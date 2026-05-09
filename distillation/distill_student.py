import json
from pathlib import Path
from typing import Final

from datasets import Dataset
from transformers import TokenizersBackend

OUTPUT_DIR: Final[str] = "./distilled_student"


def load_dataset_from_json(path: str, tokenizer: TokenizersBackend) -> Dataset:
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
