"""C's same-representation gates and Stage D's frozen-selection boundary."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.training.multihead_regression import evaluate_event_level

class CDReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("cd_reporting_tests",Path(__file__).resolve().parents[1]/"scripts/multilabel/147_report_round2_stage_d.py")
        cls.d=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.d)

    def test_same_representation_gate_and_weak_tie(self):
        g={n:{"passed":False} for n in tuple(self.d.v.C_COEFFICIENTS)[:4]}
        g['B0_EVENT-B0_W']['passed']=True
        self.assertIsNone(self.d.v.select_loss('C',g,{'C_EVENT':.9,'C_EVENT_WEAK':.9}))
        g['C_EVENT-F_C']['passed']=True;g['C_EVENT_WEAK-F_C']['passed']=True
        self.assertEqual(self.d.v.select_loss('C',g,{'C_EVENT':.9,'C_EVENT_WEAK':.9000005}),'C_EVENT_WEAK')
        self.assertEqual(self.d.v.select_loss('C',g,{'C_EVENT':.9,'C_EVENT_WEAK':.90001}),'C_EVENT')

    def test_test_reader_requires_freeze_and_checks_saved_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            with self.assertRaises(ValueError):self.d.read_leaf(p,'test')
            rng=np.random.default_rng(1)
            a={'event_id':np.arange(6).astype(str),'subject_id':np.asarray(['a']*3+['b']*3),'day_id':np.arange(6).astype(str),
                'target':rng.normal(size=(6,11)).astype(np.float32),'prediction':rng.normal(size=(6,11)).astype(np.float32)}
            metric=evaluate_event_level(a,np.ones((1,11),np.float32))
            (p/'metrics.json').write_text(json.dumps({'status':'ok','protocol':'cross_day','train_target_std':[1.]*11,'test':metric}))
            np.savez(p/'event_predictions.npz',label_names=np.asarray(LABEL_NAMES),**{'test_'+k:v for k,v in a.items()},val_prediction=np.asarray([None],dtype=object))
            _,_,score=self.d.read_leaf(p,'test',selection_frozen=True,expected_count=6)
            self.assertEqual(score.shape,(11,5))
            metric['per_label'][LABEL_NAMES[0]]['raw_r']+=.1
            (p/'metrics.json').write_text(json.dumps({'status':'ok','protocol':'cross_day','train_target_std':[1.]*11,'test':metric}))
            with self.assertRaises(ValueError):self.d.read_leaf(p,'test',selection_frozen=True,expected_count=6)

    def test_interaction_coefficients_and_36_unique_cells(self):
        self.assertEqual(len(set(self.d.ROUTES))*len(self.d.SEEDS),36)
        for name in ('event_loss_interaction','weak_loss_interaction','input_adaptation_interaction'):
            co=np.asarray(self.d.COEFFICIENTS[name])
            self.assertEqual(co.sum(),0)
            self.assertEqual(int(np.count_nonzero(co)),4)
        self.assertEqual(dict(zip(self.d.ROUTES,self.d.COEFFICIENTS['event_loss_interaction']))['B0_EVENT'],-1)

if __name__=='__main__':unittest.main()
