import json
from pathlib import Path
from typing import Final

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend

from config import STUDENT_MODEL


OUTPUT_DIR: Final[str] = "./distilled_student"


def load_dataset_from_jsonl(path: str, tokenizer: TokenizersBackend) -> Dataset:
    rows = []
    with Path.open(Path(path)) as f:
        for line in f:
            ex = json.loads(line)
            messages = [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["teacher_solution"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            rows.append({"text": text})
    return Dataset.from_list(rows)


def main():
    print(f"Loading student: {STUDENT_MODEL}")
    tokenizer = get_tokenizer(STUDENT_MODEL)
    print(f"Loaded tokenizer: {tokenizer}")
    model = get_model(STUDENT_MODEL)
    print(f"Loaded model: {model}")
    print(f"Model device: {model.device}")


def get_tokenizer(model_name: str) -> TokenizersBackend:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )


if __name__ == "__main__":
    main()
