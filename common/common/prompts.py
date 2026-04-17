from typing import Dict

def load_template(model_key: str) -> str:
    """
    Load the answer prompt template for a given model key. If the model key is
    not found in the ``ANSWER_PROMPT_TEMPLATES`` mapping, returns the "default"
    template.

    Parameters
    ----------
    model_key: str
        The key identifying the model (e.g., "llama2-7b-chat", "Hermes-btlm-3b-8k").

    Returns
    -------
    str
        The answer prompt template associated with the given model key, or the
        default template if the key is not found.
    """
    ANSWER_PROMPT_TEMPLATES: Dict[str, str] = {
    "llama2-7b-chat": "[INST] {question} [/INST]\nAnswer: (",
    "Hermes-btlm-3b-8k": "{question}\nAnswer: (",
    "Hermes-RWKV-v5-7B-HF": "\x16user\n{question}\x17\n\x16assistant\nAnswer: (",
    "Hermes-mamba-2.8b-slimpj": "\x16user\n{question}\x17\n\x16assistant\nAnswer: (",

    "default": "{question}\nAnswer: (",
    }

    return ANSWER_PROMPT_TEMPLATES.get(model_key, ANSWER_PROMPT_TEMPLATES["default"])