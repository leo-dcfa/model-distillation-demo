"""
Part 2, alternative method: ON-POLICY distillation (GKD-style).

The core idea fixes a subtle but real problem with the previous two methods.

Sequence-level and token-level distillation both train the student on prefixes
the *teacher* generated. At inference time, the student generates its own
(messier) prefixes — and has never been trained to recover from its own mistakes.
This is "exposure bias" or "train/test distribution mismatch."

On-policy distillation flips this around:
  1. The student generates a continuation from a prompt
  2. The teacher scores its predicted distribution at each position of that continuation
  3. The student is trained to match the teacher's distribution given the
     student's own (often imperfect) prefixes

It's like a coach correcting the student's mistakes in real-time, rather than
demonstrating perfect technique on its own.

We follow the GKD (Generalized Knowledge Distillation, Agarwal et al. 2024)
recipe: mix on-policy steps with off-policy steps for stability. Pure on-policy
can be unstable early when the student's generations are gibberish.

Run: python distill_student_onpolicy.py
Output: ./distilled_student_onpolicy/  (LoRA adapter)
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.config import DEVICE, EPOCHS, STUDENT_MODEL, TEACHER_DATA, TEACHER_MODEL
from src.constants import SYSTEM_PROMPT
from src.logger import MetricsLogger
from src.utils import get_model, get_tokenizer


OUTPUT_DIR = "./distilled_student_onpolicy"

# Hyperparameters
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 1e-4  # lower than off-policy — on-policy training is noisier
WARMUP_RATIO = 0.05
TEMPERATURE = 1.0  # we use reverse KL with T=1, following MiniLLM/GKD
MAX_PROMPT_LEN = 512
MAX_NEW_TOKENS = 384
ON_POLICY_RATIO = 0.5  # fraction of steps that are on-policy; rest are off-policy
GEN_TEMP = 1.0  # sampling temperature for student rollouts


# ---------- Data ----------
class PromptDataset(Dataset):
    """For on-policy training we only need the prompts — the student generates the rest."""

    def __init__(self, path, tokenizer, max_len=MAX_PROMPT_LEN):
        self.examples = []
        with Path.open(Path(path)) as f:
            for line in f:
                ex = json.loads(line)
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["question"]},
                ]
                prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)["input_ids"]
                if len(prompt_ids) > max_len:
                    continue
                self.examples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "teacher_response": ex["teacher_solution"],  # used for off-policy steps
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def pad_left(seqs, pad_id):
    """Left-pad sequences to the same length. Left-pad is required for batched generation."""
    max_len = max(len(s) for s in seqs)
    padded, mask = [], []
    for s in seqs:
        pad = max_len - len(s)
        padded.append([pad_id] * pad + s)
        mask.append([0] * pad + [1] * len(s))
    return torch.tensor(padded), torch.tensor(mask)


def pad_right(seqs, pad_id):
    """Right-pad sequences. Used for the full prompt+response sequence during loss computation."""
    max_len = max(len(s) for s in seqs)
    padded, mask = [], []
    for s in seqs:
        pad = max_len - len(s)
        padded.append(s + [pad_id] * pad)
        mask.append([1] * len(s) + [0] * pad)
    return torch.tensor(padded), torch.tensor(mask)


def reverse_kl(student_logits, teacher_logits, response_mask, T=1.0):  # noqa: N803
    """KL(student || teacher), masked to the response portion only."""
    s_logits = student_logits[:, :-1, :].contiguous() / T
    t_logits = teacher_logits[:, :-1, :].contiguous() / T
    mask = response_mask[:, 1:].contiguous().float()

    s_log = F.log_softmax(s_logits, dim=-1)
    t_log = F.log_softmax(t_logits, dim=-1)
    s_p = s_log.exp()

    # KL(s || t) = sum_v s * (log s - log t)
    kl = (s_p * (s_log - t_log)).sum(dim=-1)
    n = mask.sum().clamp(min=1)
    return (kl * mask).sum() / n


# ---------- Generation (student rollouts) ----------
@torch.no_grad()
def student_generate(student, tokenizer, prompt_batch, device, max_new_tokens=MAX_NEW_TOKENS):
    """Have the student generate continuations for a batch of prompts."""
    student.eval()
    input_ids, attn_mask = pad_left([p["prompt_ids"] for p in prompt_batch], tokenizer.pad_token_id)
    input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
    out = student.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=GEN_TEMP,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    student.train()

    # Strip left padding and split prompt vs response per example
    results = []
    for i, p in enumerate(prompt_batch):
        prompt_len = len(p["prompt_ids"])
        # Find where actual prompt starts (skip left padding)
        full = out[i].tolist()
        # The prompt occupies positions [pad_len..pad_len+prompt_len)
        pad_len = (attn_mask[i] == 0).sum().item()
        prompt_tokens = full[pad_len : pad_len + prompt_len]
        response_tokens = full[pad_len + prompt_len :]
        # Trim trailing pads from response
        while response_tokens and response_tokens[-1] == tokenizer.pad_token_id:
            response_tokens.pop()
        results.append(
            {
                "prompt_ids": prompt_tokens,
                "response_ids": response_tokens,
            }
        )
    return results


def build_loss_batch(rollouts_or_targets, tokenizer, device):
    """Concatenate prompt+response into full sequences and build response masks."""
    full_seqs, prompt_lens = [], []
    for r in rollouts_or_targets:
        full = r["prompt_ids"] + r["response_ids"]
        full_seqs.append(full)
        prompt_lens.append(len(r["prompt_ids"]))

    input_ids, attn_mask = pad_right(full_seqs, tokenizer.pad_token_id)
    # response_mask: 1 on response tokens, 0 on prompt and padding
    response_mask = torch.zeros_like(input_ids)
    for i, plen in enumerate(prompt_lens):
        end = (attn_mask[i] == 1).sum().item()
        response_mask[i, plen:end] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "response_mask": response_mask.to(device),
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--quick", action="store_true", help="Fast demo mode: epochs=1 (on-policy is the slowest method)")
    args = parser.parse_args()

    if args.quick and args.epochs == EPOCHS:
        args.epochs = 1
        print("[--quick] Using epochs=1 for fast demo mode")
    epochs = args.epochs

    print(f"Device: {DEVICE}")

    tokenizer = get_tokenizer(STUDENT_MODEL)

    print(f"Loading teacher: {TEACHER_MODEL}")
    teacher = get_model(TEACHER_MODEL)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Loading student: {STUDENT_MODEL}")
    student = get_model(STUDENT_MODEL)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    print("Loading prompt dataset...")
    dataset = PromptDataset(TEACHER_DATA, tokenizer)
    print(f"Examples: {len(dataset)}")

    # Custom batching — we don't use DataLoader's collate because batches go through
    # both a generation path (on-policy) and a teacher-data path (off-policy).
    indices = list(range(len(dataset)))

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    n_batches = len(dataset) // BATCH_SIZE
    total_steps = (n_batches * epochs) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger = MetricsLogger(run_name="onpolicy")

    # ---------- Training loop ----------
    print(f"\nStarting on-policy distillation training (on-policy ratio: {ON_POLICY_RATIO})...\n")
    student.train()
    step = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        random.shuffle(indices)
        running_loss = 0.0
        n_on = n_off = 0

        pbar = tqdm(range(0, len(indices) - BATCH_SIZE + 1, BATCH_SIZE), desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, start in enumerate(pbar):
            prompt_batch = [dataset[i] for i in indices[start : start + BATCH_SIZE]]

            # Decide on-policy vs off-policy for this microbatch
            on_policy = random.random() < ON_POLICY_RATIO  # noqa: S311

            if on_policy:
                # Student generates its own continuation
                rollouts = student_generate(student, tokenizer, prompt_batch, DEVICE)
                batch = build_loss_batch(rollouts, tokenizer, DEVICE)
                n_on += 1
            else:
                # Use the teacher's pre-generated response (teaches the student
                # what the teacher would say on familiar prefixes)
                targets = []
                for p in prompt_batch:
                    response_ids = tokenizer.encode(p["teacher_response"], add_special_tokens=False)
                    targets.append({"prompt_ids": p["prompt_ids"], "response_ids": response_ids})
                batch = build_loss_batch(targets, tokenizer, DEVICE)
                n_off += 1

            # Forward through both models on the chosen sequence
            with torch.no_grad():
                t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            s_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()

            # Reverse KL on the response portion only
            loss = reverse_kl(s_logits, t_logits, batch["response_mask"], T=TEMPERATURE)

            (loss / GRAD_ACCUM).backward()
            running_loss = 0.95 * running_loss + 0.05 * loss.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                logger.log(
                    step=step, epoch=epoch, loss=running_loss, lr=scheduler.get_last_lr()[0], on_policy_steps=n_on, off_policy_steps=n_off
                )

            pbar.set_postfix(
                loss=f"{running_loss:.3f}",
                on=n_on,
                off=n_off,
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )

        print(f"Epoch {epoch + 1}: {n_on} on-policy steps, {n_off} off-policy steps")

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved on-policy distilled student adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
