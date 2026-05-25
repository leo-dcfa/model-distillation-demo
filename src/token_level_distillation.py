import argparse
import json
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as F  # noqa: N812
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TokenizersBackend, get_cosine_schedule_with_warmup

from src.config import DEVICE, STUDENT_MODEL, TEACHER_DATA, TEACHER_MODEL
from src.logger import MetricsLogger
from src.utils import get_model, get_tokenizer


TEMPERATURE = 2.0
MAX_LENGTH = 1024
EPOCHS = 3

BATCH_SIZE = 2  # small batches since we have two models loaded in memory
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.03
OUTPUT_DIR: Final[str] = "./distilled_token_level"


class TeacherDataset(Dataset):
    def __init__(self, path: str, tokenizer: TokenizersBackend, max_length=MAX_LENGTH):
        self.examples = []
        with Path.open(Path(path)) as f:
            for line in f:
                ex = json.loads(line)
                prompt_msgs = [{"role": "user", "content": ex["question"]}]
                full_msgs = [*prompt_msgs, {"role": "assistant", "content": ex["teacher_solution"]}]
                #  the user's question only, with a "now it's the assistant's turn" marker appended (add_generation_prompt=True)
                prompt_ids = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)["input_ids"]
                #  the user's question and the teacher's full response
                full_ids = tokenizer.apply_chat_template(full_msgs)["input_ids"]
                # input ids is the full tokenized conversation (user prompt + assistant solution)
                # prompt_len is the length of the prompt portion only
                self.examples.append(
                    {
                        "input_ids": full_ids,
                        "prompt_len": len(prompt_ids),
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        return self.examples[key]


def collate(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn_mask, loss_mask = [], [], []
    for b in batch:
        ids = b["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attn_mask.append([1] * len(ids) + [0] * pad)
        m = [0] * b["prompt_len"] + [1] * (len(ids) - b["prompt_len"]) + [0] * pad
        loss_mask.append(m)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn_mask),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
    }


def kl_loss(student_logits, teacher_logits, loss_mask, T, direction="forward"):  # noqa: N803
    """Token-wise KL divergence, masked to assistant-only positions.

    student_logits, teacher_logits: (B, L, V)
    loss_mask: (B, L)  — 1 where we want loss, 0 elsewhere

    Note on shifting: the model's logits at position t predict token t+1.
    The loss mask is aligned to *predicted* tokens, so we shift logits left by one.
    """
    # Shift so logits[:, t] predicts token at position t+1
    s_logits = student_logits[:, :-1, :].contiguous()
    t_logits = teacher_logits[:, :-1, :].contiguous()
    mask = loss_mask[:, 1:].contiguous()  # mask aligned to predicted positions

    # Apply temperature
    s_log_probs = F.log_softmax(s_logits / T, dim=-1)
    t_log_probs = F.log_softmax(t_logits / T, dim=-1)
    s_probs = s_log_probs.exp()
    t_probs = t_log_probs.exp()

    if direction == "forward":
        # KL(teacher || student) = sum_v t * (log t - log s)
        kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    elif direction == "reverse":
        # KL(student || teacher) = sum_v s * (log s - log t)
        kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    else:
        raise ValueError(f"unknown KL direction: {direction}")

    # Mask and average over assistant tokens only.
    masked_kl = kl * mask
    n_tokens = mask.sum().clamp(min=1)
    return (masked_kl.sum() / n_tokens) * (T * T)


def ce_loss(student_logits, input_ids, loss_mask):
    """Standard next-token CE on assistant tokens only — anchors the student to ground truth."""
    s_logits = student_logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    mask = loss_mask[:, 1:].contiguous()
    flat_logits = s_logits.view(-1, s_logits.size(-1))
    flat_targets = targets.view(-1)
    flat_mask = mask.view(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (losses * flat_mask).sum() / flat_mask.sum().clamp(min=1)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kl", choices=["forward", "reverse"], default="forward", help="KL direction. forward=mode-covering, reverse=mode-seeking."
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight on CE loss; (1-alpha) goes to KL distillation loss.")
    parser.add_argument("--quick", action="store_true", help="Fast demo mode: epochs=2")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"KL direction: {args.kl}, alpha (CE weight): {args.alpha}")

    # Tokenizer (shared between teacher and student because they're same family)
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

    # Verify tokenizer compatibility — token-level distillation requires the same vocab.
    teacher_vocab = teacher.config.vocab_size
    student_vocab = student.config.vocab_size
    assert teacher_vocab == student_vocab, (
        f"Tokenizer vocab mismatch ({teacher_vocab} vs {student_vocab}). Token-level distillation requires matching tokenizers."
    )

    # Data
    print("Loading dataset...")
    dataset = TeacherDataset(TEACHER_DATA, tokenizer)
    print(f"Examples after filtering: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    total_steps = (len(loader) * EPOCHS) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger = MetricsLogger(run_name=f"logits_{args.kl}")

    # ---------- Training loop ----------
    print("\nStarting token-level distillation training...")
    student.train()
    step = 0
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        running_kl = running_ce = 0.0
        for i, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Teacher forward (no grad)
            with torch.no_grad():
                t_out = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                t_logits = t_out.logits.float()  # promote to fp32 for stable softmax

            # Student forward (grads through LoRA only)
            s_out = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            s_logits = s_out.logits.float()

            # Compute losses
            l_kl = kl_loss(s_logits, t_logits, batch["loss_mask"], T=TEMPERATURE, direction=args.kl)
            l_ce = ce_loss(s_logits, batch["input_ids"], batch["loss_mask"]) if args.alpha > 0 else torch.tensor(0.0, device=DEVICE)
            loss = args.alpha * l_ce + (1.0 - args.alpha) * l_kl

            (loss / GRAD_ACCUM).backward()
            running_kl = 0.95 * running_kl + 0.05 * l_kl.item()
            running_ce = 0.95 * running_ce + 0.05 * l_ce.item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                logger.log(
                    step=step,
                    epoch=epoch,
                    loss=running_kl + args.alpha * running_ce,
                    kl=running_kl,
                    ce=running_ce,
                    lr=scheduler.get_last_lr()[0],
                )

            pbar.set_postfix(kl=f"{running_kl:.3f}", ce=f"{running_ce:.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved token-level distilled student adapter to {OUTPUT_DIR}")
    print(f"Method: token-level KL ({args.kl}), alpha={args.alpha}, T={TEMPERATURE}")


if __name__ == "__main__":
    main()
