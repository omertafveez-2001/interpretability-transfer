"""Utility helpers for loading causal language models."""

from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

load_dotenv()

MODEL_REPO_MAP = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "Hermes-RWKV-v5-7B-HF": "EleutherAI/Hermes-RWKV-v5-7B-HF",
    "Hermes-btlm-3b-8k": "EleutherAI/Hermes-btlm-3b-8k",
    "Hermes-mamba-2.8b-slimpj": "EleutherAI/Hermes-mamba-2.8b-slimpj",
}


def load_model(
    model_key: str, *, device_map: str | None = "auto", **kwargs
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a tokenizer and causal LM for one of the supported model identifiers.

    Parameters
    ----------
    model_key: str
        One of the keys in ``MODEL_REPO_MAP`` (e.g., "mamba-2.8", "BTLM-3b",
        "llama2-7b-chat", "RWKV-v5-7b").
    device_map: str | None, optional
        Passed to ``AutoModelForCausalLM.from_pretrained``. Defaults to ``"auto"``
        so models place themselves on available GPUs/CPUs. Set ``None`` to use
        transformers defaults.
    **kwargs:
        Extra keyword args forwarded to both tokenizer and model loading calls
        (e.g., ``revision``, ``torch_dtype``).

    Returns
    -------
    (tokenizer, model)
        A tuple containing the tokenizer and the model objects.

    Raises
    ------
    ValueError
        If an unsupported model key is provided.
    """

    if model_key not in MODEL_REPO_MAP:
        raise ValueError(
            f"Unsupported model_key '{model_key}'. Choose one of: {list(MODEL_REPO_MAP)}"
        )

    repo_id = MODEL_REPO_MAP[model_key]
    HF_token = os.getenv("HF_TOKEN")

    # Mamba models require the mamba-ssm package for efficient kernels.
    if "mamba" in model_key:
        try:
            import mamba_ssm
            tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            trust_remote_code=True,
            token=HF_token,
            )
            model = MambaLMHeadModel.from_pretrained("EleutherAI/Hermes-mamba-2.8b-slimpj")

            return tokenizer, model
        except ImportError as exc:
            raise ImportError(
                "mamba-ssm is required for Mamba models. Install with "
                "`pip install mamba-ssm` (CUDA) or refer to the project instructions."
            ) from exc
    
    if "RWKV" in model_key:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, token=HF_token, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            token=HF_token,
            torch_dtype="auto",
            # don't use device_map="auto" — RWKV needs to be fully on one device
        ).cuda()
        return tokenizer, model

    # Tokenizers sometimes require remote code for bespoke architectures too.
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, token=HF_token, use_fast=False, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map=device_map,
        trust_remote_code=True,
        token=HF_token,
        **kwargs,
    )

    return tokenizer, model


LOGIT_LENS_MODELS = {
    "mamba-790m":  "state-spaces/mamba-790m",
    "mamba-1.4b":  "state-spaces/mamba-1.4b",
    "mamba-2.8b":  "state-spaces/mamba-2.8b",
    "rwkv-v4-3b":  "RWKV/rwkv-4-3b-pile",
    "btlm-3b":     "cerebras/btlm-3b-8k-base",
}

def load_lens_model(
    model_key: str,
    *,
    torch_dtype: str = "auto",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a tokenizer and model for the logit lens experiment.
 
    Mamba is loaded via mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel
    for speed. Pythia and RWKV load via AutoModelForCausalLM.
    RWKV is loaded without device_map to avoid accelerate meta device issues.
 
    Parameters
    ----------
    model_key : str
        One of the keys in LOGIT_LENS_MODELS.
    torch_dtype : str
        Dtype for model weights.
 
    Returns
    -------
    (tokenizer, model)
    """
    if model_key not in LOGIT_LENS_MODELS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Choose from: {list(LOGIT_LENS_MODELS)}"
        )
 
    repo_id  = LOGIT_LENS_MODELS[model_key]
    hf_token = os.getenv("HF_TOKEN")
 
    if "mamba" in model_key:
        # Mamba uses GPT-NeoX tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", trust_remote_code=True, token=hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # mamba_ssm loads directly to GPU
        model = MambaLMHeadModel.from_pretrained(repo_id).cuda()
 
    elif "rwkv" in model_key:
        # RWKV-v4 pile models use the GPT-NeoX 20B tokenizer (same as Mamba)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", trust_remote_code=True, token=hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            token=hf_token,
        ).cuda()
 
    else:
        # Pythia and other standard HF models
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=True, token=hf_token, use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            token=hf_token,
        )
 
    model.eval()
    return tokenizer, model
 
 
