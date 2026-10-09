# Training configs

`configs/4_train/<model>/` holds one config per method for `llada` (LLaDA-8B-Instruct), `llada15` (LLaDA-1.5) and `dream` (Dream-v0-Instruct-7B). The eight configs of a model differ only in the objective and the fairness term:

| Config | Objective | Fairness term | Groups | λ |
|---|---|---|---|---|
| `sft.yaml` | SFT | — | — | — |
| `sft_log_ratio.yaml` | SFT | log-ratio | option pair | 1 |
| `sft_js_div.yaml` | SFT | JS div. | option pair + lexicon | 30 |
| `sft_logprob_var.yaml` | SFT | log-prob. var. | option pair + lexicon | 1 |
| `dopsd.yaml` | dOPSD | — | — | — |
| `dopsd_log_ratio.yaml` | dOPSD | log-ratio | option pair | 1 |
| `dopsd_js_div.yaml` | dOPSD | JS div. | option pair + lexicon | 30 |
| `dopsd_logprob_var.yaml` | dOPSD | log-prob. var. | option pair + lexicon | 1 |

Every config trains a LoRA adapter (r = 32 on all attention and MLP projections) for 938 steps, 3 epochs of `data/openbbq/train-10k` at an effective batch size of 32, with learning rate 4e-5 and seed 1234, and saves it to `outputs/<model>/<config>/`.

Any field can be overridden on the command line after the config, e.g.

```bash
bash scripts/4_train.sh configs/4_train/llada/dopsd_logprob_var.yaml --max_steps 300 --run_config llada/dopsd_300
```

The lexicon-based configs read `data/lexicon/<tokenizer>/regrouped/lexicon.json` from step 3; `llada15` uses LLaDA's lexicon. The settings left at dOPSD's defaults are listed as comments at the end of each file.
