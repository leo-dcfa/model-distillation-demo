import argparse
import bisect
import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.config import DEVICE, EPOCHS, STUDENT_MODEL, TEACHER_DATA
from src.logger import MetricsLogger
from src.utils import get_model, get_tokenizer


# Cross-tokenizer distillation deliberately picks a teacher from a *different*
# model family from the student so the two vocabularies don't match — that's
# the whole point of the method. Hence a separate constant rather than reusing
# config.TEACHER_MODEL (which assumes a same-family teacher).
CROSS_TEACHER_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR = "./distilled_student_uld"

BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.03
TEMPERATURE = 2.0
TOP_K = 50  # how many sorted probabilities to match. Higher = more signal but more noise.
MAX_LEN = 1024


# ---------- Cross-tokenizer alignment ----------
# Both models tokenize the same text differently. We align positions via character
# offsets so that "teacher position t" and "student position s" predict tokens
# that *end at the same character* in the response text.


class AlignedDataset(Dataset):
    """For each example, store both tokenizations and their character offset maps."""

    def __init__(self, path, teacher_tok, student_tok, max_len=MAX_LEN):
        self.examples = []
        with Path.open(Path(path)) as f:
            for line in f:
                ex = json.loads(line)
                response = ex["teacher_solution"]

                # Build the "input" text seen by each model. We treat distillation
                # as matching distributions over the response text — the prompt is
                # context for both models, but the loss only fires on response tokens.
                t_prompt = teacher_tok.apply_chat_template(
                    [{"role": "user", "content": ex["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                s_prompt = student_tok.apply_chat_template(
                    [{"role": "user", "content": ex["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Tokenize each (prompt + response) with the corresponding tokenizer,
                # capturing offsets so we know where each token lives in the response text.
                t_enc = teacher_tok(t_prompt + response, return_offsets_mapping=True, add_special_tokens=False)
                s_enc = student_tok(s_prompt + response, return_offsets_mapping=True, add_special_tokens=False)

                if len(t_enc["input_ids"]) > max_len or len(s_enc["input_ids"]) > max_len:
                    continue

                self.examples.append(
                    {
                        "teacher_ids": t_enc["input_ids"],
                        "teacher_offsets": t_enc["offset_mapping"],
                        "teacher_resp_start_char": len(t_prompt),
                        "student_ids": s_enc["input_ids"],
                        "student_offsets": s_enc["offset_mapping"],
                        "student_resp_start_char": len(s_prompt),
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def build_alignment(student_offsets, student_resp_start, teacher_offsets, teacher_resp_start):
    """For each student response token, find the matching teacher position.

    Matching rule: for student token s ending at character c (in the response text),
    pick the teacher token whose ending character is closest to c. Both measured
    relative to the start of the response text.

    Returns a list of (student_pos, teacher_pos) pairs.
    """
    student_response_tokens = [
        (i, end - student_resp_start) for i, (start, end) in enumerate(student_offsets) if start >= student_resp_start and end > start
    ]
    teacher_response_tokens = [
        (i, end - teacher_resp_start) for i, (start, end) in enumerate(teacher_offsets) if start >= teacher_resp_start and end > start
    ]

    if not teacher_response_tokens:
        return []

    teacher_ends = [e for _, e in teacher_response_tokens]
    pairs = []
    for s_pos, s_end in student_response_tokens:
        idx = bisect.bisect_left(teacher_ends, s_end)
        idx = max(0, min(idx, len(teacher_response_tokens) - 1))
        candidates = [idx]
        if idx > 0:
            candidates.append(idx - 1)
        best = min(candidates, key=lambda i: abs(teacher_ends[i] - s_end))
        pairs.append((s_pos, teacher_response_tokens[best][0]))
    return pairs


def collate(batch, teacher_pad, student_pad):
    """Pad teacher and student sequences independently. Carry alignment lists per example."""
    t_max = max(len(b["teacher_ids"]) for b in batch)
    s_max = max(len(b["student_ids"]) for b in batch)

    t_ids, t_attn = [], []
    s_ids, s_attn = [], []
    alignments = []

    for b in batch:
        t_pad = t_max - len(b["teacher_ids"])
        s_pad = s_max - len(b["student_ids"])
        t_ids.append(b["teacher_ids"] + [teacher_pad] * t_pad)
        t_attn.append([1] * len(b["teacher_ids"]) + [0] * t_pad)
        s_ids.append(b["student_ids"] + [student_pad] * s_pad)
        s_attn.append([1] * len(b["student_ids"]) + [0] * s_pad)

        alignments.append(
            build_alignment(
                b["student_offsets"],
                b["student_resp_start_char"],
                b["teacher_offsets"],
                b["teacher_resp_start_char"],
            )
        )

    return {
        "teacher_ids": torch.tensor(t_ids),
        "teacher_attn": torch.tensor(t_attn),
        "student_ids": torch.tensor(s_ids),
        "student_attn": torch.tensor(s_attn),
        "alignments": alignments,  # list of lists of (s_pos, t_pos) tuples
    }


# ---------- ULD loss: KL between sorted top-K distributions ----------
def uld_kl_loss(student_logits, teacher_logits, alignments, top_k, T):  # noqa: N803
    """Compute KL between sorted top-K teacher and student distributions at aligned positions.

    student_logits: (B, S_len, V_s)
    teacher_logits: (B, T_len, V_t)
    alignments[b]: list of (s_pos, t_pos) for batch element b

    Note on shifting: model logits at position t predict token t+1. So to predict
    the token at student position s_pos, we read student_logits[:, s_pos - 1].
    Same for teacher. We skip alignments where either index would be 0.
    """
    losses = []
    for b in range(student_logits.size(0)):
        s_indices = [s for (s, _) in alignments[b] if s > 0]
        t_indices = [t for (s, t) in alignments[b] if s > 0]
        if not s_indices:
            continue

        s_pos = torch.tensor(s_indices, device=student_logits.device) - 1
        t_pos = torch.tensor(t_indices, device=teacher_logits.device) - 1

        s_l = student_logits[b].index_select(0, s_pos) / T  # (n_aligned, V_s)
        t_l = teacher_logits[b].index_select(0, t_pos) / T  # (n_aligned, V_t)

        # Take top-K from each side and renormalize as a K-way distribution.
        # The sorted top-K probability vectors now live in a common K-dim space.
        s_top = s_l.topk(top_k, dim=-1).values
        t_top = t_l.topk(top_k, dim=-1).values

        s_log_probs = F.log_softmax(s_top, dim=-1)
        t_log_probs = F.log_softmax(t_top, dim=-1)
        t_probs = t_log_probs.exp()

        # Forward KL(teacher || student) on the sorted distributions.
        kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        losses.append(kl.mean() * T * T)

    if not losses:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--quick", action="store_true", help="Fast demo mode: epochs=1")
    args = parser.parse_args()

    if args.quick and args.epochs == EPOCHS:
        args.epochs = 1
        print("[--quick] Using epochs=1 for fast demo mode")
    epochs = args.epochs

    print(f"Device: {DEVICE}")

    teacher_tok = get_tokenizer(CROSS_TEACHER_MODEL)
    student_tok = get_tokenizer(STUDENT_MODEL)

    print(f"Teacher tokenizer: vocab size {teacher_tok.vocab_size}")
    print(f"Student tokenizer: vocab size {student_tok.vocab_size}")
    print("(They differ — this is the whole point of cross-tokenizer distillation.)")

    print(f"\nLoading teacher: {CROSS_TEACHER_MODEL}")
    teacher = get_model(CROSS_TEACHER_MODEL)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Loading student: {STUDENT_MODEL}")
    student = get_model(STUDENT_MODEL)
    student = get_peft_model(
        student,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    student.print_trainable_parameters()

    print("Loading aligned dataset...")
    dataset = AlignedDataset(TEACHER_DATA, teacher_tok, student_tok)
    print(f"Examples after filtering: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate(b, teacher_tok.pad_token_id, student_tok.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    total_steps = (len(loader) * epochs) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger = MetricsLogger(run_name="uld")

    print("\nStarting cross-tokenizer (ULD) distillation training...\n")
    student.train()
    step = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(pbar):
            with torch.no_grad():
                t_logits = teacher(
                    input_ids=batch["teacher_ids"].to(DEVICE),
                    attention_mask=batch["teacher_attn"].to(DEVICE),
                ).logits.float()

            s_logits = student(
                input_ids=batch["student_ids"].to(DEVICE),
                attention_mask=batch["student_attn"].to(DEVICE),
            ).logits.float()

            loss = uld_kl_loss(s_logits, t_logits, batch["alignments"], TOP_K, TEMPERATURE)
            (loss / GRAD_ACCUM).backward()
            running = 0.95 * running + 0.05 * loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                logger.log(step=step, epoch=epoch, loss=running, lr=scheduler.get_last_lr()[0])

            pbar.set_postfix(loss=f"{running:.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUTPUT_DIR)
    student_tok.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved cross-tokenizer distilled student adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
