from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model_registry import MODEL_REPO_MAP, load_model

RNN_MODEL_KEYS = {
    "RWKV-v5-7b",
    "Hermes-RWKV-v5-7B-HF",
    "Hermes-mamba-2.8b-slimpj",
    "mamba-2.8",
}


@dataclass
class SteeringVector:
    """
    Load a causal LM and compute / apply activation steering vectors.

    Two construction paths:

      # Via short model key (see MODEL_REPO_MAP in model_loader.py)
      sv = SteeringVector.from_model_key("llama2-7b-chat")

      # Via any HuggingFace repo id or local path
      sv = SteeringVector.from_hf_path("gpt2")

    Then:
      vectors = sv.compute_steering_vectors(pos_prompts, neg_prompts)
      output  = sv.steered_generate("Hello!", vectors, multiplier=20.0)
    """

    model:     AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device:    torch.device = field(default_factory=lambda: torch.device("cpu"))
    is_rnn:    bool         = False


    @classmethod
    def from_model_key(
        cls,
        model_key: str,
        *,
        adapter_path: Optional[str] = None,
        device_map: str | None = "auto",
        **load_kwargs,
    ) -> "SteeringVector":
        """
        Load via a short model key defined in MODEL_REPO_MAP.
        e.g. "llama2-7b-chat", "mamba-2.8", "RWKV-v5-7b".
        """
        # load the model using model_key from load_model from common.model_registry
        tokenizer, model = load_model(model_key, device_map=device_map, **load_kwargs)
        is_rnn = model_key in RNN_MODEL_KEYS
        return cls._finalise(tokenizer, model, adapter_path, is_rnn=is_rnn)

    @classmethod
    def from_hf_path(
        cls,
        hf_path: str,
        *,
        adapter_path: Optional[str] = None,
        device_map: str | None = "auto",
        torch_dtype: torch.dtype = torch.float16,
        is_rnn: Optional[bool] = False,
        **hf_kwargs,
    ) -> "SteeringVector":

        """
        Load via any HuggingFace repo id or local directory path.
        e.g. "gpt2", "mistralai/Mistral-7B-v0.1", "/path/to/model".
        """

        if "mamba" in hf_path.lower():
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
            model     = MambaLMHeadModel.from_pretrained(
                hf_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **hf_kwargs,
            )
            return cls._finalise(tokenizer, model, adapter_path, is_rnn=True)

        # load tokenizer and model directly using HuggingFace's Auto classes
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=False, **hf_kwargs)
        model     = AutoModelForCausalLM.from_pretrained(
            hf_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **hf_kwargs,
        )
        return cls._finalise(tokenizer, model, adapter_path, is_rnn=is_rnn)

    @classmethod
    def _finalise(
        cls,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        adapter_path: Optional[str],
        is_rnn: Optional[bool] = False,
    ) -> "SteeringVector":

        """Shared post-load setup: pad token, eval mode, optional PEFT adapter."""
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()

        # Optionally load a PEFT adapter on top of the base model.
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("pip install peft to use adapter_path") from exc
            model = PeftModel.from_pretrained(model, adapter_path)

        # only move to cuda manually if accelerate hasn't already dispatched the model
        # (accelerate sets hf_device_map when device_map="auto" is used)
        if not hasattr(model, "hf_device_map"):
            model.cuda()

        device = next(model.parameters()).device
        return cls(model=model, tokenizer=tokenizer, device=device, is_rnn=is_rnn)
        
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _layer_names(self) -> List[str]:

        """
        This function attempts to infer the names of the transformer layers in the model.
        It will return a list of strings that can be used to access the layers.
        Returns all the layers in the model.
        """

        # Heuristic layer name inference for common causal LM architectures.
        m = self.model

        # If self.model does not work, try other architectures such as self.blocks, self.transformer.h
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return [f"model.layers.{i}" for i in range(len(m.model.layers))]

        # GPT-2, BTLM (Cerebras), GPT-NeoX style
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return [f"transformer.h.{i}" for i in range(len(m.transformer.h))]

        # RWKV-v5 (EleutherAI/Hermes-RWKV-v5-*)
        if hasattr(m, "rwkv") and hasattr(m.rwkv, "blocks"):
            return [f"rwkv.blocks.{i}.attention" for i in range(len(m.rwkv.blocks))]

        # Mamba (state-spaces/mamba-* and EleutherAI/Hermes-mamba-*)
        if hasattr(m, "backbone") and hasattr(m.backbone, "layers"):
            return [f"backbone.layers.{i}" for i in range(len(m.backbone.layers))]

        # Fallback: scan top-level attributes for any list that looks like blocks
        for attr in ["layers", "blocks", "h"]:
            if hasattr(m, attr):
                block_list = getattr(m, attr)
                return [f"{attr}.{i}" for i in range(len(block_list))]

        raise RuntimeError(
            f"Cannot infer layer names for {type(m).__name__}. "
            "Pass layer_names explicitly to the relevant method."
        )

    # These are Transformer style hooks
    @contextmanager
    def _collect_hooks(self, store: Dict[str, torch.Tensor], layer_names: List[str]):
        """Attach read-only forward hooks; remove them on exit.
        Parameters:
            - store: dict to populate with activations {layer_name: tensor(batch, hidden_dim)}
            - layer_names: list of layer names to attach hooks to
        """
        handles = []
        for name in layer_names:
            # get the module corresponding to the layer name (e.g. "model.layers.0")
            module = self.model.get_submodule(name)

            def _hook(_, __, output, _name=name):

                # output can be a tensor or a tuple (hidden, ...); we want the hidden state tensor
                # if it's a tuple or list, we take the first element as the hidden state; otherwise we use the output directly
                out = output[0] if isinstance(output, (tuple, list)) else output
                
                # take the last token's hidden state (assuming output shape is (batch, seq_len, hidden_dim)) and store it in the provided dict, detached from the graph and moved to CPU
                store[_name] = out[:, -1, :].detach().cpu() # Shape: (batch, hidden_dim)
               

            # register the hook and keep the handle for later removal
            handles.append(module.register_forward_hook(_hook))
        try:
            # yield control back to the caller, allowing them to run a forward pass with the hooks active
            yield
        finally:
            # remove all hooks to clean up
            for h in handles:
                h.remove()

    @contextmanager
    def _steer_hooks(self, vectors: Dict[str, torch.Tensor], multiplier: float):
        """Inject multiplier * vector into the residual stream at each layer.
        Parameters:
            - vectors: dict of steering vectors {layer_name: tensor(hidden_dim,)}
            - multiplier: float
        """

        handles = []
        for name, vec in vectors.items():
            # get the module corresponding to the layer name (e.g. "model.layers.0")
            module = self.model.get_submodule(name)
            # get the change to apply to the hidden state: multiplier * vector, moved to the correct device
            delta  = (multiplier * vec).to(self.device)

            def _hook(_, __, output, _delta=delta):
                if isinstance(output, (tuple, list)):
                    # broadcast _delta to match the batch size and add it to the hidden state (assumed to be the first element of the output tuple)
                    # output[0] -> (batch, seq_len, hidden_dim)
                    hidden = output[0] + _delta.unsqueeze(0).unsqueeze(0)
                    return (hidden,) + output[1:]
                return output + _delta.unsqueeze(0).unsqueeze(0)

            handles.append(module.register_forward_hook(_hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def resolve_layer_indices(self, indices: List[int]) -> List[str]:
        """Map integer layer indices to architecture-specific layer name strings.

        Uses the same heuristics as ``_layer_names()`` so the mapping is always
        consistent with what ``compute_steering_vectors`` would use.

        Parameters
        ----------
        indices : list[int]
            Zero-based layer indices, e.g. ``[10, 12, 15, 16]``.

        Returns
        -------
        list[str]
            Resolved layer name strings, e.g. ``["model.layers.10", ...]``.

        Raises
        ------
        IndexError
            If any index falls outside the model's layer count.
        """
        all_names = self._layer_names()
        resolved = []
        for i in indices:
            if i < 0 or i >= len(all_names):
                raise IndexError(
                    f"Layer index {i} is out of range — model has {len(all_names)} layers (0–{len(all_names)-1})."
                )
            resolved.append(all_names[i])
        return resolved

    def get_hidden_states(
        self,
        prompt: str,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward pass → mean hidden state per layer.
        Returns {layer_name: tensor(1, hidden_dim)}.
        """

        # If layer_names is not provided, infer them using the _layer_names method
        names  = layer_names or self._layer_names()
        store: Dict[str, torch.Tensor] = {}

        if self.is_rnn and hasattr(self.model, "backbone"):
            # Mamba doesn't accept attention_mask
            inputs = {"input_ids": self._tokenize(prompt)["input_ids"]}
        else:
            inputs = self._tokenize(prompt)
            if self.is_rnn:
                # RWKV doesn't use attention_mask
                inputs.pop("attention_mask", None)

        with self._collect_hooks(store, names), torch.no_grad():
            self.model(**inputs)

        return store

    def compute_steering_vectors(
        self,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str],
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        act_l = E[h_l | z] - E[h_l | ¬z]

        Averages hidden states over all prompt pairs before subtracting.
        Returns {layer_name: tensor(hidden_dim)}.
        """

        # I do not want a skewed number of positive vs negative prompts, so I will assert that they are of equal length
        assert len(positive_prompts) == len(negative_prompts), \
            "positive and negative prompt lists must have equal length."

        names = layer_names or self._layer_names()

        def _mean_acts(prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
            accum: Dict[str, torch.Tensor] = {}
            for prompt in prompts:
                for k, v in self.get_hidden_states(prompt, layer_names=names).items():
                    accum[k] = accum.get(k, torch.zeros_like(v)) + v
            # compute mean vector per layer.
            return {k: v / len(prompts) for k, v in accum.items()}

        pos_mean = _mean_acts(positive_prompts)
        neg_mean = _mean_acts(negative_prompts)

        # return steering vector per layer: difference of means, squeezed to remove batch dimension
        return {
            name: (pos_mean[name] - neg_mean[name]).squeeze(0)
            for name in pos_mean
            if name in neg_mean
        }

    def _generate_rwkv_stateful(
        self,
        input_ids: torch.Tensor,
        steering_vectors: Dict[str, torch.Tensor],
        multiplier: float,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Stateful greedy generation for HF RWKV models.

        HF's model.generate() passes past_key_values between steps, but RWKV
        uses a separate `state` argument — so generate() never threads the
        recurrent state through, and the model reprocesses the full sequence
        at every step (O(n²)).  This loop fixes that by calling forward()
        directly and passing `state` explicitly, making generation O(n).
        """
        generated: List[int] = []
        state = None

        with self._steer_hooks(steering_vectors, multiplier), torch.no_grad():
            # Prefill: process all prompt tokens in one parallel pass.
            out   = self.model(input_ids=input_ids, use_cache=True)
            state = out.state
            next_token = out.logits[0, -1, :].argmax(dim=-1).reshape(1, 1)
            generated.append(next_token.item())

            # Decode: one token at a time, carrying the recurrent state forward.
            for _ in range(max_new_tokens - 1):
                out   = self.model(input_ids=next_token, state=state, use_cache=True)
                state = out.state
                next_token = out.logits[0, -1, :].argmax(dim=-1).reshape(1, 1)
                generated.append(next_token.item())

        prompt_ids = input_ids[0].tolist()
        return torch.tensor([prompt_ids + generated], dtype=torch.long, device=self.device)

    def steered_generate(
    self,
    prompt: str,
    steering_vectors: Dict[str, torch.Tensor],
    *,
    multiplier: float = 20.0,
    max_new_tokens: int = 200,
    **generate_kwargs,
    ) -> str:
        """
        Generate text with steering vectors injected at each layer.
        multiplier > 0 amplifies behavior z; < 0 suppresses it; 0 = baseline.
        """

        if self.is_rnn and hasattr(self.model, "backbone"):
            # Mamba doesn't accept attention_mask
            inputs = {"input_ids": self._tokenize(prompt)["input_ids"]}
            # mamba_ssm's generate() uses max_length not max_new_tokens
            max_length = inputs["input_ids"].shape[1] + max_new_tokens
            gen_kwargs = {"max_length": max_length, **generate_kwargs}
            with self._steer_hooks(steering_vectors, multiplier), torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        elif self.is_rnn and hasattr(self.model, "rwkv"):
            # RWKV: use stateful generation to avoid O(n²) cost from HF's
            # generate() not threading the recurrent state between steps.
            inputs = self._tokenize(prompt)
            inputs.pop("attention_mask", None)
            output_ids = self._generate_rwkv_stateful(
                inputs["input_ids"], steering_vectors, multiplier, max_new_tokens
            )

        else:
            inputs   = self._tokenize(prompt)
            gen_kwargs = {"max_new_tokens": max_new_tokens, **generate_kwargs}
            with self._steer_hooks(steering_vectors, multiplier), torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    def save_vectors(self, vectors: Dict[str, torch.Tensor], path: str | Path) -> None:
        torch.save(vectors, path)
        print(f"Saved steering vectors → {path}")

    def load_vectors(self, path: str | Path) -> Dict[str, torch.Tensor]:
        return torch.load(path, map_location="cpu")