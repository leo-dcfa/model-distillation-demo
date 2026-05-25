import json
from pathlib import Path
from typing import Final, cast

import torch
from datasets.load import load_dataset
from tqdm import tqdm

from src.config import DEVICE, TEACHER_MODEL
from src.constants import DEFAULT_NUMBER_EXAMPLES, SYSTEM_PROMPT
from src.utils import get_model, get_tokenizer


DEFAULT_OUTPUT_TEACHER_DATA: Final[str] = "teacher_data.jsonl"


def main():
    print(f"Loading teacher {TEACHER_MODEL}")
    tokenizer = get_tokenizer(TEACHER_MODEL)
    model = get_model(TEACHER_MODEL)
    # run .eval to indicate we are running inference
    model.eval()
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(DEFAULT_NUMBER_EXAMPLES))
    print(f"Generating {len(ds)} solutions with the teacher...")

    with Path.open(Path(DEFAULT_OUTPUT_TEACHER_DATA), "w") as f:
        for ex in tqdm(ds):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["question"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = cast("str", tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))

                f.write(
                    json.dumps(
                        {
                            "question": ex["question"],
                            "ground_truth": ex["answer"],
                            "teacher_solution": response.strip(),
                        }
                    )
                    + "\n"
                )
    print(f"Saved {DEFAULT_NUMBER_EXAMPLES} examples to {DEFAULT_OUTPUT_TEACHER_DATA}")


if __name__ == "__main__":
    main()
