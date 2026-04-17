# Interpretability Transfer

Replication of [_Does Transformer Interpretability Transfer to RNNs?_](https://arxiv.org/abs/2312.01939) (EleutherAI). Applies three interpretability techniques — logit lens, tuned lens, and steering vectors — to both transformer and non-transformer architectures (Mamba, RWKV).

## Models

| Model | Architecture |
|---|---|
| LLaMA-2-7b-chat | Transformer |
| BTLM-3b | Transformer |
| Mamba-790m / 1.4b / 2.8b | State-space model |
| Hermes-Mamba-2.8b | State-space model |
| RWKV-v4-3b | RNN |
| Hermes-RWKV-v5-7B | RNN |

## Repository Structure

```
common/           # Shared model loading, logit lens, and steering vector utilities
datagen/          # Scripts to download and preprocess evaluation data
logit-lens/       # Logit lens evaluation pipeline
steering-vectors/ # Steering vector computation and evaluation
tuned-lens/       # Forked tuned-lens library with Mamba and RWKV support (see below)
graphs/           # Generated plots
utils.py          # Plotting utilities
playground.ipynb  # Interactive analysis notebook
```

## Forked: `tuned-lens`

`tuned-lens/` is a fork of [AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens) with the following changes to support non-transformer architectures:

**`tuned_lens/model_surgery.py`**
- Added `MambaLMHeadModel` to the union of supported model types
- `get_unembedding_matrix()`: reads from Mamba's `lm_head`
- `get_final_norm()`: reads from Mamba's `backbone.norm_f` and RWKV's `ln_out`
- `get_transformer_layers()`: recognises Mamba's `backbone.layers` and RWKV's `rwkv.blocks`

**`tuned_lens/scripts/ingredients.py`**
- Mamba models are loaded via `MambaLMHeadModel.from_pretrained()` (mamba_ssm), never with `device_map="auto"`
- RWKV models are loaded via `AutoModelForCausalLM` but also without `device_map="auto"` (must fit on a single device)
- Mamba and RWKV use the EleutherAI GPT-NeoX tokenizer regardless of model config

**`tuned_lens/scripts/eval_loop.py`**
- HF transformer models use `output_hidden_states=True` to collect intermediate representations
- Mamba models register forward hooks on `backbone.embedding` and each `backbone.layers[i]` because `mamba_ssm` does not return hidden states in the standard HF format

## Running Experiments

**Logit Lens**
```bash
cd logit-lens
./run_logit_lens.sh
```

**Steering Vectors**
```bash
cd steering-vectors
./run_steering_pipeline.sh
```

**Tuned Lens (train)**
```bash
cd tuned-lens
./run_tuned_lens.sh
```

**Tuned Lens (eval)**
```bash
cd tuned-lens
./run_tuned_lens_eval.sh
```

## Installation

```bash
pip install transformers accelerate mamba-ssm datasets pandas matplotlib seaborn protobuf bitsandbytes
```

## Known Issues

- **BTLM**: Built against an older version of `transformers` — downgrade with `pip install "transformers==4.37.2"`
- **Mamba CUDA extension**: If `mamba_ssm` crashes on loading `selective_scan_cuda`, install the CUDA 12 runtime: `pip install nvidia-cuda-runtime-cu12`
- **RWKV**: Requires `protobuf` and `bitsandbytes` to be installed
