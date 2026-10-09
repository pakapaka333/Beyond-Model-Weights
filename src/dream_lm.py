"""lm-eval wrapper for Dream, over the `diffllm` model of dOPSD's lm-eval fork."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DREAM_EVAL_DIR = REPO_ROOT / "third_party" / "dOPSD" / "eval" / "Dream" / "eval_instruct"
# DiffLLM.generate_until imports the fork's top-level `model` package.
if str(DREAM_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(DREAM_EVAL_DIR))

from lm_eval.models.diffllm import DiffLLM  # noqa: E402


class DreamLM(DiffLLM):
    def __init__(self, *args, model_path=None, adapter=None, self_chat_template=False,
                 prompt_prefix="", prompt_suffix="", **kwargs):
        if model_path is not None:
            kwargs.setdefault("pretrained", model_path)
        # Dream's tokenizer adds no special tokens either way.
        kwargs.pop("add_special_tokens", None)
        super().__init__(*args, **kwargs)
        self.self_chat_template = bool(self_chat_template)
        self.prompt_prefix = prompt_prefix or ""
        self.prompt_suffix = prompt_suffix or ""
        self.total_nfe = 0
        if adapter:
            from peft import PeftModel
            path = Path(adapter)
            if not path.is_absolute():
                path = REPO_ROOT / path
            self.model = PeftModel.from_pretrained(self.model, str(path))
            self.model.eval()

    def chat_template(self, chat_template: bool = False):
        return self.tokenizer.chat_template if chat_template else None

    def _generate_batch(self, prompts):
        if self.self_chat_template:
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{self.prompt_prefix}{p}{self.prompt_suffix}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        responses, nfe = super()._generate_batch(prompts)
        self.total_nfe += int(nfe)
        return responses, nfe
