"""Source-space, physical-support and matched-availability regression tests."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

SCRIPTS=Path(os.environ.get('MAE_SCRIPT_ROOT',Path(__file__).resolve().parents[1]/'scripts/multilabel'))


def load(filename,name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


class Round1Tests(unittest.TestCase):
    def test_cuda_attention_retry_updates_once_and_preserves_later_rng(self):
        import torch
        from daily_multimodal.training import modality_mae as mae
        if not torch.cuda.is_available():self.skipTest('CUDA numerical recovery requires the GPU experiment runtime')
        source=np.random.default_rng(7).normal(size=(8,2000,59)).astype(np.float32)
        runtime=mae.MAERuntime(epochs=1,batch_size=4,device='cuda',health_probe_count=4,min_relative_variation=0)
        original_guard=mae.guarded_optimizer_step;original_step=torch.optim.AdamW.step
        expected_rng=[];updates=0;fired=False
        def injected_guard(model,optimizer,loss,context):
            nonlocal fired,expected_rng
            if not fired:
                fired=True;expected_rng=torch.cuda.get_rng_state_all()
                raise RuntimeError('non-finite gradients: injected CUDA backend failure')
            return original_guard(model,optimizer,loss,context)
        def counted_step(optimizer,*args,**kwargs):
            nonlocal updates
            updates+=1
            return original_step(optimizer,*args,**kwargs)
        with patch.object(mae,'guarded_optimizer_step',side_effect=injected_guard),patch.object(torch.optim.AdamW,'step',new=counted_step):
            _,audit=mae.train_eeg_mae(source,np.arange(4),np.arange(4,8),embedding_dim=8,encoder_layers=1,decoder_layers=1,heads=2,runtime=runtime,retry_attention_math=True)
        self.assertEqual(updates,1);self.assertEqual(audit['optimizer_steps'],1)
        self.assertEqual(len(audit['attention_retry_records']),1)
        self.assertEqual(audit['attention_retry_records'][0]['rows_skipped'],0)
        self.assertEqual(audit['attention_retry_records'][0]['row_count'],4)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected_rng,torch.cuda.get_rng_state_all())))

    def test_sorted_source_reader_matches_seek_pixels(self):
        try:
            import cv2
        except ModuleNotFoundError as exc:
            if exc.name!='cv2':raise
            self.skipTest('OpenCV decoder runs on the ncc video host')
        video=load('134_build_round1_video_inputs.py','round1_video_reader_test')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),24,(128,128))
            self.assertTrue(writer.isOpened())
            rng=np.random.default_rng(5)
            for _ in range(260):writer.write(rng.integers(0,256,(128,128,3),dtype=np.uint8))
            writer.release();cap=cv2.VideoCapture(str(path))
            indices=[1,9,22,40,55,230,240,250]
            actual=video.read_sorted_frames(cap,indices+indices[::-1],24);cap.release()
            cap=cv2.VideoCapture(str(path))
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,img=cap.read();self.assertTrue(ok)
                expected=cv2.cvtColor(cv2.resize(img,(112,112),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                np.testing.assert_array_equal(actual[fi],expected)
            cap.release()

    def test_eeg_epoch_resume_keeps_rng_optimizer_and_selector(self):
        import torch
        from daily_multimodal.training import modality_mae as mae
        source=np.random.default_rng(1).normal(size=(8,2000,59)).astype(np.float32)
        runtime=mae.MAERuntime(epochs=2,batch_size=4,device='cpu',health_probe_count=4,min_relative_variation=0)
        kwargs=dict(embedding_dim=8,encoder_layers=1,decoder_layers=1,heads=2,runtime=runtime)
        reference,audit=mae.train_eeg_mae(source,np.arange(4),np.arange(4,8),**kwargs)
        original=mae._evaluate_eeg;calls=0
        def interrupted(*args,**kw):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('simulated execution interruption')
            return original(*args,**kw)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recovery.pt'
            with patch.object(mae,'_evaluate_eeg',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'simulated execution interruption'):
                    mae.train_eeg_mae(source,np.arange(4),np.arange(4,8),recovery_path=path,**kwargs)
            resumed,actual=mae.train_eeg_mae(source,np.arange(4),np.arange(4,8),recovery_path=path,**kwargs)
        self.assertEqual(actual['history'],audit['history'])
        self.assertEqual(actual['best_epoch'],audit['best_epoch'])
        for key,value in reference.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[key]))

    def test_half_open_signal_support(self):
        audit=load('133_build_round1_eeg_inputs.py','round1_audit')
        self.assertFalse(audit.overlaps(0,[(10,20)]))
        self.assertTrue(audit.overlaps(2,[(10,20)]))
        self.assertEqual(audit.duration([(0,10),(5,15),(20,30)]),25)

    def test_local_roi_centers(self):
        video=load('134_build_round1_video_inputs.py','round1_video_inputs')
        np.testing.assert_array_equal(video.sample_frames(16),[1,3,5,7,9,11,13,15])
        with self.assertRaises(ValueError):video.sample_frames(7)

    def test_source_row_collision_does_not_alias(self):
        runner=load('112_run_modality_mae.py','round1_eeg_runner')
        canonical=np.full((2,2000,59),3,dtype=np.float32);raw=np.full_like(canonical,9)
        joined=runner.JoinedEEGSource(canonical,raw,[{'source_kind':'raw','source_row':0},{'source_kind':'canonical','source_row':0}])
        out=joined[np.asarray([1,0,1])]
        np.testing.assert_array_equal(out[:,0,0],[3,9,3])
        np.testing.assert_array_equal(canonical[:,0,0],[3,3])

    def test_paired_bootstrap_preserves_event_membership(self):
        report=load('136_summarize_round1_inputs.py','round1_summary_test')
        y=np.arange(33,dtype=float).reshape(3,11)
        values={'val_target':y,'val_prediction':y+.1,'val_event_id':np.asarray(['e1','e2','e3']),
                'val_subject_id':np.asarray(['s1','s1','s2']),'val_day_id':np.asarray(['d1','d2','d1'])}
        result=report.bootstrap(values,values,'val',np.ones(11),20,13)
        self.assertTrue(all(value=={'low':0.,'high':0.} for value in result.values()))
        changed={**values,'val_event_id':np.asarray(['e2','e1','e3'])}
        with self.assertRaises(ValueError):report.bootstrap(values,changed,'val',np.ones(11),20,13)

    def test_common_mask_only_modifies_video(self):
        runner=load('118_run_mae_mt11_event_ablation.py','round1_downstream')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ids=np.asarray([f'eeg_{i:06d}' for i in range(28819)])
            payload={'sample_id_matrix':ids[:23].reshape(1,23),'tokens':np.ones((1,23,4,256),dtype=np.float32),
                     'modality_mask':np.ones((1,23,4),dtype=np.int8),'source_npz_json':np.asarray(json.dumps({'video_A1':'original'})),
                     'train_index':np.asarray([0]),'val_index':np.asarray([],dtype=int),'test_index':np.asarray([],dtype=int),
                     'event_id':np.asarray(['event-0'])}
            np.savez(root/'reference.npz',**payload);mask=np.ones(28819,dtype=bool);mask[3]=False
            np.savez(root/'common.npz',sample_id=ids,valid_mask=mask)
            runner._write_replaced_bag(reference=root/'reference.npz',mae_paths={},destination=root/'candidate.npz',condition='B0_VIDEO_COMMON',video_common_mask=root/'common.npz')
            with np.load(root/'candidate.npz') as z:
                for slot in (0,1,3):np.testing.assert_array_equal(z['tokens'][:,:,slot],payload['tokens'][:,:,slot])
                np.testing.assert_array_equal(z['train_index'],payload['train_index'])
                self.assertFalse(z['modality_mask'][0,3,2]);self.assertTrue((z['tokens'][0,3,2]==0).all())


if __name__=='__main__':unittest.main()
