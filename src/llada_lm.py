"""lm-eval wrapper for LLaDA, over dOPSD's eval/LLaDA/eval_llada.py."""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
LLADA_DIR = REPO_ROOT / "third_party" / "dOPSD" / "eval" / "LLaDA"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

from eval_llada import LLaDAEvalHarness  # noqa: E402
from generate import (  # noqa: E402
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
)


class LLaDALM(LLaDAEvalHarness):
    def __init__(self, *args, temperature=0.0, add_special_tokens=True, adapter=None,
                 self_chat_template=False, prompt_prefix="", prompt_suffix="", **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)
        self.add_special_tokens = bool(add_special_tokens)
        # With self_chat_template, the rendered context (wrapped in prefix/suffix) is
        # placed inside the user turn, as dOPSD's own GSM8K evaluation does.
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

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def chat_template(self, chat_template: bool = False):
        return self.tokenizer.chat_template if chat_template else None

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    def _sampler(self):
        if not self.use_cache:
            return generate
        return generate_with_dual_cache if self.dual_cache else generate_with_prefix_cache

    def generate_until(self, requests):
        sampler = self._sampler()
        out = []
        for req in tqdm(requests, desc="Generating..."):
            context, gen_kwargs = req.args
            if self.self_chat_template:
                context = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{self.prompt_prefix}{context}{self.prompt_suffix}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            input_ids = self.tokenizer(context, add_special_tokens=self.add_special_tokens)["input_ids"]
            prompt = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            generated, nfe = sampler(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                remasking=self.remasking,
                mask_id=self.mask_id,
                threshold=self.threshold,
                logits_eos_inf=self.logits_eos_inf,
                confidence_eos_eot_inf=self.confidence_eos_eot_inf,
            )
            self.total_nfe += int(nfe)

            text = self.tokenizer.decode(generated[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in gen_kwargs.get("until") or []:
                if stop_seq in text:
                    text = text.split(stop_seq)[0]
            text = self.tokenizer.decode(self.tokenizer(text)["input_ids"], skip_special_tokens=True)
            out.append(text)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        return out
