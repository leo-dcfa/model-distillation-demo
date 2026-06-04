"""Load distilled (LoRA) students into TransformerLens for interpretability.

Each ``distilled_*`` folder is a LoRA adapter over Qwen2.5-0.5B that only touches
the attention projections (q/k/v/o — see the ``adapter_config.json`` files). To
look inside one we merge the adapter into a fresh copy of the base model and hand
the merged HuggingFace model to TransformerLens, which gives us hooks into the
attention patterns, the residual stream, and the unembedding.

The processing flags (``fold_ln`` / ``center_writing_weights`` / ``center_unembed``)
are TransformerLens' defaults and are what make logit-lens and residual-stream
attribution numerically meaningful, so we keep them on.
"""

import torch
from peft import PeftModel
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from src.config import DEVICE, STUDENT_MODEL
from src.utils import get_tokenizer


# (label, adapter_dir) — adapter_dir is None for the un-tuned base student.
MODELS: list[tuple[str, str | None]] = [
    ("base", None),
    ("sequence_level", "distilled_sequence_level"),
    ("token_level", "distilled_token_level"),
    ("on_policy", "distilled_student_onpolicy"),
    ("uld", "distilled_student_uld"),
]


def load_hooked(adapter: str | None = None, *, device: str = DEVICE, dtype: torch.dtype = torch.float32) -> HookedTransformer:
    """Merge a LoRA adapter into the base student and wrap it in a HookedTransformer.

    Pass ``adapter=None`` (or ``"base"``) to load the un-tuned student. We start
    from a fresh base model each call because ``merge_and_unload`` rewrites the
    weights in place.
    """
    hf_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, dtype=dtype)
    if adapter and adapter != "base":
        hf_model = PeftModel.from_pretrained(hf_model, adapter).merge_and_unload()

    model = HookedTransformer.from_pretrained(
        STUDENT_MODEL,
        hf_model=hf_model,
        tokenizer=get_tokenizer(STUDENT_MODEL),
        device=device,
        dtype=dtype,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    model.eval()
    return model


def free(model: HookedTransformer) -> None:
    """Drop a model and reclaim its GPU memory before loading the next one."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
