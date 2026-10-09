"""Semi-autoregressive block decoding for Dream.

Dream's released sampler has no block length. scripts/1_setup.sh fetches its
generation_utils.py and applies patches/dream_generation_utils.patch, which adds
a `block_length` argument to `_sample`; `install` binds that `_sample` to a model
instance with a fixed block length.
"""

import importlib.util
import types
from pathlib import Path

PATCHED = Path(__file__).resolve().parents[1] / "third_party" / "dream" / "generation_utils.py"


def _patched_module():
    if not PATCHED.exists():
        raise FileNotFoundError(f"{PATCHED} is missing; run scripts/1_setup.sh first.")
    spec = importlib.util.spec_from_file_location("dream_generation_utils_blocked", PATCHED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(model, block_length: int) -> None:
    """Make `model.diffusion_generate()` decode in blocks of `block_length` tokens."""
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    sample = _patched_module().DreamGenerationMixin._sample

    def _sample(self, *args, **kwargs):
        return sample(self, *args, block_length=block_length, **kwargs)

    target._sample = types.MethodType(_sample, target)
