import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend

from src.config import DEVICE


def get_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )


def get_tokenizer(model_name: str) -> TokenizersBackend:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not isinstance(tokenizer, TokenizersBackend):
        raise TypeError(f"Cannot proceed; expected Tonenizer backend, got {type(tokenizer)}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
