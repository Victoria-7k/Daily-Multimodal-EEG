from __future__ import annotations

import importlib.util
import inspect
import sys


def main() -> int:
    print(f"python={sys.version}")
    for name in [
        "numpy",
        "scipy",
        "torch",
        "braindecode",
        "safetensors",
        "huggingface_hub",
        "mne",
        "einops",
        "linear_attention_transformer",
        "rotary_embedding_torch",
    ]:
        spec = importlib.util.find_spec(name)
        print(f"{name}={'FOUND' if spec else 'MISSING'}")
    try:
        import torch

        print(f"torch_version={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
    except Exception as exc:
        print(f"torch_error={type(exc).__name__}: {exc}")
    try:
        from braindecode.models import EEGPT

        print(f"EEGPT={EEGPT}")
        print(f"EEGPT_signature={inspect.signature(EEGPT)}")
    except Exception as exc:
        print(f"EEGPT_error={type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
