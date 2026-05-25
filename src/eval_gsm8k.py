"""Evaluate distilled student adapters on held-out GSM8K problems.

For each adapter, generate solutions to N held-out problems and compute final-
answer accuracy by parsing the '#### <number>' line in both the model output
and the ground-truth solution.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets.load import load_dataset
from peft import PeftModel
from tqdm import tqdm

from src.config import DEVICE, STUDENT_MODEL
from src.constants import SYSTEM_PROMPT
from src.utils import get_model, get_tokenizer


# Match either '#### N' (GSM8K convention requested in our system prompt) or
# '\boxed{N}' (the default format Qwen2.5-3B-Instruct produces when no system
# prompt is given — the KL-trained adapters fall back to this because their
# training data didn't carry the system prompt through).
ANSWER_RE = re.compile(r"(?:####\s*|\\boxed\{)(-?[\d,]+(?:\.\d+)?)")
MAX_NEW_TOKENS = 512


def parse_answer(text: str) -> str | None:
    """Extract the final '#### N' answer, normalising commas and trailing zeros."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").rstrip(".")
    try:
        f = float(raw)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def evaluate(model, tokenizer, examples) -> tuple[int, int]:
    correct = 0
    for ex in tqdm(examples, desc="eval"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        pred = parse_answer(response)
        truth = parse_answer(ex["answer"])
        if pred is not None and pred == truth:
            correct += 1
    return correct, len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=[
            "distilled_sequence_level",
            "distilled_token_level",
            "distilled_student_onpolicy",
            "distilled_student_uld",
        ],
        help="Adapter directories to evaluate. Pass 'base' to also score the un-tuned student.",
    )
    parser.add_argument("--n", type=int, default=50, help="Held-out examples to evaluate.")
    parser.add_argument("--out", default="eval_results.jsonl", help="Per-adapter results JSONL.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading {args.n} held-out GSM8K test examples...")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))

    tokenizer = get_tokenizer(STUDENT_MODEL)

    results = []
    for adapter in args.adapters:
        print(f"\n=== {adapter} ===")
        model = get_model(STUDENT_MODEL)
        if adapter != "base":
            if not Path(adapter).exists():
                print("  skipping — directory not found")
                continue
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        correct, total = evaluate(model, tokenizer, ds)
        acc = correct / total
        print(f"  {adapter}: {correct}/{total} = {acc:.1%}")
        results.append({"adapter": adapter, "correct": correct, "total": total, "accuracy": acc})

        # Free GPU memory before loading next adapter
        del model
        torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['adapter']:36s}  {r['accuracy']:.1%}  ({r['correct']}/{r['total']})")

    with Path(args.out).open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
