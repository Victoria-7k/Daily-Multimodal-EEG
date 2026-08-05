from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import warnings
from pathlib import Path


STANDARD_1020_NAMES = [
    "FP1",
    "FPZ",
    "FP2",
    "AF7",
    "AF3",
    "AFZ",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POZ",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2",
]


def main() -> int:
    print("stage=install_mne_stub", flush=True)
    _install_mne_stub()
    print("stage=load_eegpt_module", flush=True)
    eegpt = _load_eegpt_module()
    print("stage=module_loaded", flush=True)
    EEGPT = eegpt.EEGPT
    print(f"EEGPT={EEGPT}", flush=True)
    print(f"signature={inspect.signature(EEGPT)}", flush=True)
    print("stage=construct_model", flush=True)
    model = EEGPT(
        n_outputs=1,
        n_chans=59,
        n_times=2500,
        sfreq=250,
        chs_info=_fake_chs_info(59),
        return_encoder_output=True,
    )
    print(f"model_class={type(model).__name__}", flush=True)
    print(f"param_count={sum(p.numel() for p in model.parameters())}", flush=True)
    return 0


def _install_mne_stub() -> None:
    mne = types.ModuleType("mne")
    channels = types.ModuleType("mne.channels")
    utils = types.ModuleType("mne.utils")

    class Montage:
        ch_names = STANDARD_1020_NAMES

        def get_positions(self):
            return {"ch_pos": {name: [0.0, 0.0, 0.0] for name in self.ch_names}}

    def make_standard_montage(name: str) -> Montage:
        if name != "standard_1020":
            raise ValueError(f"unsupported montage {name}")
        return Montage()

    def _soft_import(name: str, purpose: str, strict: bool = False):
        try:
            return __import__(name)
        except Exception:
            if strict:
                raise
            return False

    def warn(message: str, *args, **kwargs) -> None:
        warnings.warn(message, stacklevel=2)

    channels.make_standard_montage = make_standard_montage
    utils._soft_import = _soft_import
    utils.warn = warn
    mne.channels = channels
    mne.utils = utils
    sys.modules["mne"] = mne
    sys.modules["mne.channels"] = channels
    sys.modules["mne.utils"] = utils


def _load_eegpt_module():
    import site

    print("stage=find_braindecode_package", flush=True)
    candidates = []
    for base in site.getsitepackages():
        candidates.append(Path(base) / "braindecode")
    candidates.append(Path(sys.prefix) / "lib/python3.11/site-packages/braindecode")
    package_root = next(path for path in candidates if (path / "models/eegpt.py").is_file())

    bd_pkg = types.ModuleType("braindecode")
    bd_pkg.__path__ = [str(package_root)]
    sys.modules["braindecode"] = bd_pkg
    print("stage=install_braindecode_util_stub", flush=True)
    _install_braindecode_util_stub()
    print("stage=install_braindecode_modules_stub", flush=True)
    _install_braindecode_modules_stub()

    models_pkg = types.ModuleType("braindecode.models")
    models_pkg.__path__ = [str(package_root / "models")]
    sys.modules["braindecode.models"] = models_pkg
    _install_interpolated_model_stub()

    spec = importlib.util.spec_from_file_location(
        "braindecode.models.eegpt",
        package_root / "models/eegpt.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to locate braindecode.models.eegpt")
    module = importlib.util.module_from_spec(spec)
    sys.modules["braindecode.models.eegpt"] = module
    print("stage=exec_eegpt_module", flush=True)
    spec.loader.exec_module(module)
    return module


def _fake_chs_info(n_chans: int) -> list[dict[str, object]]:
    names = STANDARD_1020_NAMES[:n_chans]
    return [{"ch_name": name, "kind": 2, "loc": [0.0] * 12} for name in names]


def _install_braindecode_util_stub() -> None:
    import numpy as np
    import torch

    util = types.ModuleType("braindecode.util")

    def np_to_th(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
        if not hasattr(X, "__len__"):
            X = [X]
        values = np.asarray(X)
        if dtype is not None:
            values = values.astype(dtype)
        tensor = torch.tensor(values, requires_grad=requires_grad, **tensor_kwargs)
        if pin_memory:
            tensor = tensor.pin_memory()
        return tensor

    util.np_to_th = np_to_th
    sys.modules["braindecode.util"] = util


def _install_braindecode_modules_stub() -> None:
    import torch
    import torch.nn.functional as F
    from torch import nn

    modules = types.ModuleType("braindecode.modules")
    conv = types.ModuleType("braindecode.modules.convolution")
    linear = types.ModuleType("braindecode.modules.linear")

    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            drop_prob = float(self.drop_prob or 0.0)
            if drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    class Conv1dWithConstraint(nn.Conv1d):
        def __init__(self, *args, max_norm=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_norm = max_norm

        def forward(self, input):
            weight = self.weight
            if self.max_norm is not None:
                weight = torch.renorm(weight, p=2, dim=0, maxnorm=float(self.max_norm))
            return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    class LinearWithConstraint(nn.Linear):
        def __init__(self, *args, max_norm=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_norm = max_norm

        def forward(self, input):
            weight = self.weight
            if self.max_norm is not None:
                weight = torch.renorm(weight, p=2, dim=0, maxnorm=float(self.max_norm))
            return F.linear(input, weight, self.bias)

    modules.DropPath = DropPath
    modules.Conv1dWithConstraint = Conv1dWithConstraint
    modules.LinearWithConstraint = LinearWithConstraint
    conv.Conv1dWithConstraint = Conv1dWithConstraint
    linear.LinearWithConstraint = LinearWithConstraint
    sys.modules["braindecode.modules"] = modules
    sys.modules["braindecode.modules.convolution"] = conv
    sys.modules["braindecode.modules.linear"] = linear


def _install_interpolated_model_stub() -> None:
    interpolated = types.ModuleType("braindecode.models.interpolated")

    def InterpolatedModel(model_cls, target_chs_info, name="InterpolatedModel", **kwargs):
        del target_chs_info, kwargs
        return type(str(name), (model_cls,), {})

    interpolated.InterpolatedModel = InterpolatedModel
    sys.modules["braindecode.models.interpolated"] = interpolated


if __name__ == "__main__":
    raise SystemExit(main())
