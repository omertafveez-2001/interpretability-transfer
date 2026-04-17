"""
logit_lens.py
=============
Logit lens implementation for Pythia, Mamba (mamba_ssm), and RWKV.

The logit lens projects each layer's hidden state through the final
LayerNorm and unembedding matrix W_U to get a distribution over the
vocabulary at each layer. Perplexity of the true next token under this
distribution is computed per layer, then averaged over all tokens in
the dataset to get bits-per-byte as a function of layer depth.

Formula (equation 2 in the paper):
    f(h_l) = LayerNorm[h_l] @ W_U
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model_registry import LOGIT_LENS_MODELS, load_lens_model


@dataclass
class LogitLens:
    """
    Compute logit lens bits-per-byte at each layer for a given model.

    Two construction paths:

      # Via short model key (see LOGIT_LENS_MODELS in model_registry.py)
      ll = LogitLens.from_model_key("pythia-1.4b")

      # Via already-loaded model and tokenizer
      ll = LogitLens.from_model(model, tokenizer)

    Then:
      results = ll.compute(texts)
      ll.print_results(results)
    """

    model:     object
    tokenizer: AutoTokenizer
    device:    torch.device = field(default_factory=lambda: torch.device("cuda"))


    @classmethod
    def from_model_key(cls, model_key: str) -> "LogitLens":
        """Load via a short model key defined in LOGIT_LENS_MODELS."""
        tokenizer, model = load_lens_model(model_key)
        device = cls._get_device(model)
        return cls(model=model, tokenizer=tokenizer, device=device)

    @classmethod
    def from_model(cls, model, tokenizer: AutoTokenizer) -> "LogitLens":
        """Load from an already-instantiated model and tokenizer."""
        device = cls._get_device(model)
        return cls(model=model, tokenizer=tokenizer, device=device)

    @staticmethod
    def _get_device(model) -> torch.device:
        """Get the device of the first non-meta parameter."""
        return next(
            (p.device for p in model.parameters() if p.device.type != "meta"),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    @staticmethod
    def _is_mamba_ssm(model) -> bool:
        """Check if model is mamba_ssm's MambaLMHeadModel (not HF MambaForCausalLM)."""
        return type(model).__name__ == "MambaLMHeadModel" and hasattr(model, "backbone")

    def _get_layer_names(self) -> List[str]:
        """
        Infer the list of transformer/RNN block module names for the model.
        Returns names that can be passed to model.get_submodule().
        """
        m = self.model

        # Pythia / GPT-NeoX style
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return [f"gpt_neox.layers.{i}" for i in range(len(m.gpt_neox.layers))]

        # Mamba (mamba_ssm) — backbone.layers
        if self._is_mamba_ssm(m):
            return [f"backbone.layers.{i}" for i in range(len(m.backbone.layers))]

        # LLaMA, Mistral, and most modern HF causal LMs
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return [f"model.layers.{i}" for i in range(len(m.model.layers))]

        # RWKV-v4 HF
        if hasattr(m, "rwkv") and hasattr(m.rwkv, "blocks"):
            return [f"rwkv.blocks.{i}" for i in range(len(m.rwkv.blocks))]

        # GPT-2 / BTLM style
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return [f"transformer.h.{i}" for i in range(len(m.transformer.h))]

        raise RuntimeError(
            f"Cannot infer layer names for {type(m).__name__}. "
            "Pass layer_names explicitly to compute()."
        )

    def _get_final_ln_and_unembed(self) -> Tuple[torch.nn.Module, torch.Tensor]:
        """
        Extract the final LayerNorm and unembedding matrix W_U from the model.
        Returns (layer_norm_module, W_U) where W_U has shape (hidden_dim, vocab_size).
        """
        m = self.model

        # Pythia / GPT-NeoX
        if hasattr(m, "gpt_neox"):
            ln  = m.gpt_neox.final_layer_norm
            W_U = m.embed_out.weight.T                  # (hidden, vocab)
            return ln, W_U

        # Mamba (mamba_ssm) — backbone.norm_f + lm_head
        if self._is_mamba_ssm(m):
            ln  = m.backbone.norm_f
            W_U = m.lm_head.weight.T                    # (hidden, vocab)
            return ln, W_U

        # RWKV-v4 HF
        if hasattr(m, "rwkv"):
            ln  = m.rwkv.ln_out
            W_U = m.head.weight.T                       # (hidden, vocab)
            return ln, W_U

        # LLaMA / Mistral
        if hasattr(m, "model") and hasattr(m, "lm_head"):
            ln  = m.model.norm
            W_U = m.lm_head.weight.T
            return ln, W_U

        # GPT-2 / BTLM style
        if hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"):
            ln  = m.transformer.ln_f
            W_U = m.lm_head.weight.T
            return ln, W_U

        raise RuntimeError(
            f"Cannot extract final LN and unembed for {type(m).__name__}."
        )

    def _prepare_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and move to model device, stripping unsupported keys."""
        enc    = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        # Mamba (mamba_ssm) and RWKV don't accept attention_mask
        if self._is_mamba_ssm(self.model) or hasattr(self.model, "rwkv"):
            inputs.pop("attention_mask", None)

        return inputs

    def _collect_hidden_states(
        self, text: str, layer_names: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Run a forward pass and collect the hidden state at each layer per token.

        Returns
        -------
        hidden_states : {layer_name: tensor(seq_len, hidden_dim)}
        input_ids     : tensor(1, seq_len)
        """
        inputs  = self._prepare_inputs(text)
        store:  Dict[str, torch.Tensor] = {}
        handles = []

        for name in layer_names:
            module = self.model.get_submodule(name)

            def _hook(_, __, output, _name=name):
                # take first element if tuple (hidden_states, ...), else use directly
                out = output[0] if isinstance(output, (tuple, list)) else output
                # out: (batch, seq_len, hidden_dim) — store per-token, drop batch dim
                store[_name] = out[0].detach()  # keep original dtype (fp16/bf16/fp32)

            handles.append(module.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                if self._is_mamba_ssm(self.model):
                    # mamba_ssm takes only input_ids
                    self.model(inputs["input_ids"])
                else:
                    self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        return store, inputs["input_ids"]

    def compute(
        self,
        texts: List[str],
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute logit lens bits-per-byte at each layer, averaged over all texts.

        For each layer l and each token position t:
            logits_t_l = LayerNorm(h_l[t]) @ W_U
            loss_t_l   = CrossEntropy(logits_t_l, true_next_token_t)

        Converts cross-entropy loss (nats) to bits-per-byte:
            bpb = loss / log(2) / avg_bytes_per_token

        Parameters
        ----------
        texts : List[str]
            Evaluation texts to average over.
        layer_names : List[str] | None
            Layer names to probe. Inferred automatically if not provided.

        Returns
        -------
        {layer_name: mean_bits_per_byte, "final": final_layer_bpb}
        """
        names        = layer_names or self._get_layer_names()
        ln, W_U      = self._get_final_ln_and_unembed()
        # Determine the model's native dtype from the norm's parameters so we
        # can cast hidden states to match before applying the norm, then upcast
        # to float32 for the matmul with W_U to avoid overflow.
        ln_dtype     = next(ln.parameters()).dtype
        W_U          = W_U.detach().to(self.device, dtype=torch.float32)

        accum: Dict[str, List[float]] = {name: [] for name in names}
        accum["final"] = []
        log2 = torch.log(torch.tensor(2.0))

        for text in texts:
            hidden, input_ids = self._collect_hidden_states(text, names)

            # true next tokens: shift input_ids left by 1
            target = input_ids[0, 1:].to(self.device)  # (seq_len - 1,)
            if target.shape[0] == 0:
                continue

            # compute actual bytes-per-token for this text
            n_tokens = input_ids.shape[1]
            n_bytes  = len(text.encode("utf-8"))
            bytes_per_token = n_bytes / n_tokens

            for name in names:
                h = hidden[name].to(self.device)        # (seq_len, hidden_dim)
                h = h[:-1]                              # align with targets

                with torch.no_grad():
                    h_normed = ln(h.to(ln_dtype)).float()  # norm in model dtype, then upcast
                    logits   = h_normed @ W_U              # (seq_len-1, vocab_size)

                loss = F.cross_entropy(logits, target, reduction="mean")
                bpb  = loss.item() / log2.item() / bytes_per_token
                accum[name].append(bpb)

            # final layer — use model's actual output logits
            inputs = self._prepare_inputs(text)
            with torch.no_grad():
                if self._is_mamba_ssm(self.model):
                    out = self.model(inputs["input_ids"])
                else:
                    out = self.model(**inputs)

            final_logits = out.logits[0, :-1, :].float()
            final_loss   = F.cross_entropy(final_logits, target, reduction="mean")
            accum["final"].append(final_loss.item() / log2.item() / bytes_per_token)

        return {name: sum(vals) / len(vals) for name, vals in accum.items() if vals}

    def print_results(
        self,
        results: Dict[str, float],
        model_name: str = "",
        layer_names: Optional[List[str]] = None,
    ) -> None:
        """Print logit lens results as a table with layer depth fractions."""
        names    = layer_names or self._get_layer_names()
        n_layers = len(names)

        print(f"\n{'='*55}")
        print(f"  Logit Lens{f' — {model_name}' if model_name else ''}")
        print(f"{'='*55}")
        print(f"  {'Layer':>6}  {'Depth':>8}  {'Bits/byte':>12}")
        print(f"{'─'*55}")

        for i, name in enumerate(names):
            if name in results:
                depth = (i + 1) / n_layers
                print(f"  {i:>6}  {depth:>8.3f}  {results[name]:>12.4f}")

        if "final" in results:
            print(f"{'─'*55}")
            print(f"  {'final':>6}  {'1.000':>8}  {results['final']:>12.4f}")

        print(f"{'='*55}\n")
