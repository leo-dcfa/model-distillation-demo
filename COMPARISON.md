# Distillation methods: side-by-side comparison

Each method trained for 15 epochs against 500 GSM8K teacher solutions
(`Qwen2.5-3B-Instruct` for methods 1–3, `SmolLM2-1.7B-Instruct` for ULD),
then evaluated on 100 held-out GSM8K test problems via greedy decoding.

The system prompt
(`"...end with the final answer in the form '#### <number>'"`) is now carried
through `prompt_msgs` / `full_msgs` in every training script — so the students
see at training time the exact context the eval uses, and natively produce
`#### N` instead of the model's default `\boxed{N}`.

## Headline numbers

| Method | Teacher | GSM8K@100 | Δ vs base | Train time | Loss start → end |
| --- | --- | --: | --: | --: | --: |
| **base** (no distillation) | — | 9% | — | — | — |
| **Sequence-level** (SFT on teacher text) | Qwen2.5-3B-Instruct | 43% | +34 | 2.8 min | 1.13 → 0.17 |
| **Token-level KL** (forward, α=0.1, T=2) | Qwen2.5-3B-Instruct | 37% | +28 | 5.7 min | 0.57 → 0.59 ⚠ |
| **On-policy GKD** (reverse KL, 50/50 mix) | Qwen2.5-3B-Instruct | **44%** | **+35** | **114 min** | 0.22 → 0.19 |
| **Cross-tokenizer ULD** (sorted top-K KL) | SmolLM2-1.7B-Instruct | 12% | +3 | 4.0 min | 0.13 → 0.11 |

All students are LoRA adapters (r=16, q/k/v/o projections) on `Qwen2.5-0.5B`.
Greedy decoding, 512 max new tokens. Loss values across rows aren't comparable
in magnitude (different losses, scales, temperatures) — only trajectories
within a row are meaningful.

## What changed from the 5-epoch run

| | 5ep @ n=50 | 15ep @ n=100 | delta |
| --- | --: | --: | --: |
| base | 10% | 9% | flat |
| Sequence-level | 40% | 43% | +3 |
| Token-level | 44% | **37%** | **-7** |
| On-policy | 46% | 44% | -2 |
| ULD | 6% | 12% | +6 |

Sequence-level and ULD got better with more training. **Token-level got
worse.** Its training loss confirms the diagnosis: it bottoms out at 0.185
mid-training and *climbs back to 0.594* by the end. The 0.5B student is
memorizing 500 teacher distributions long before epoch 15, then drifting.

On-policy doesn't show the same drift even at the same epoch count, because
its training data is *constantly regenerated* by the student — every
on-policy step produces fresh contexts the student hasn't seen. That's the
implicit regularization GKD buys you, beyond just closing the exposure-bias
gap.

Sequence-level resists overfitting in a different way: CE on literal tokens
is harder to overshoot than KL on full distributions. The student can
"perfectly match" a saved sequence at most once.

## Method-by-method

### Sequence-level (43%)
Cheapest and most predictable. Cleanly converging CE (1.13 → 0.17 over 15
epochs) and the best resistance to overfitting in this regime — at 15 epochs
it's still climbing where token-level is collapsing. Strong default baseline
for small datasets.

### Token-level forward KL (37%, ⚠ overfit)
The cautionary tale of this run. At 5 epochs it was the best non-on-policy
method (44%); at 15 epochs it's the worst of the same-tokenizer methods (37%)
because the loss curve U-turned around epoch 5 and climbed steadily after
that. The dense per-position signal that makes token-level powerful with a
big dataset becomes a liability with 500 examples: too many gradient steps,
too few distinct distributions to fit, the student memorizes and then drifts.

A future run with fewer epochs (3–5) or a held-out validation set for early
stopping would likely beat sequence-level here. As configured it doesn't.

### On-policy GKD (44%)
Best score, by 1 example over sequence-level (within noise on n=100). Notable
that it didn't overfit despite 15 epochs of training with rich per-token KL —
the constantly-changing on-policy data acts as implicit regularization.

The price: **114 minutes** vs 2.8–5.7 for the off-policy methods. Roughly
20–40× the compute for a tied-with-noise score. Where on-policy *should*
shine is on tasks where the student's generation distribution diverges
materially from the teacher's (long-horizon agentic tasks, compounding
errors). GSM8K with short responses isn't that regime.

### Cross-tokenizer ULD (12%)
Doubled from 6% (5ep) → 12% (15ep), and the system-prompt fix is part of why
— but it still trails the same-tokenizer methods by 25+ points. Two reasons:
1. **Weaker teacher.** SmolLM2-1.7B-Instruct (the open stand-in for the
   gated Llama-3.2-1B-Instruct) is materially worse at GSM8K than the Qwen
   3B teacher used by methods 1–3.
2. **Method losses by design.** Character-offset alignment is approximate
   (the two tokenizers split text very differently), sorted top-K KL throws
   away token identity, and the tail past K=50 is gone.

Reasonable lower bound for cross-tokenizer if you genuinely have no
same-family teacher option, but on identical data against same-tokenizer
methods it's strictly the worst tool.

## Caveats

- **n=100 is still small.** A 1–2 example gap between sequence-level and
  on-policy can flip with a different RNG seed or eval split. The clear
  signal is the *ordering* (on-policy ≈ sequence-level > token-level >> ULD),
  not the exact numbers.
- **Single seed.** No averaging.
- **Greedy decoding only.** Reverse-KL methods (on-policy) were trained to
  be mode-seeking; sampling at temperature could change the ranking.
- **500 training examples is small.** The token-level overfitting story is
  partly a function of dataset size. With 10K+ examples token-level should
  comfortably beat sequence-level, matching the literature.
- **One epoch count.** A proper comparison would track each method's eval
  accuracy *vs epoch* and report each method's peak, not just the 15-epoch
  endpoint. As reported, token-level is penalized for our choice of stopping
  too late.

## Reproducing

```bash
EPOCHS=15 uv run python -m src.sequence_level_distillation
EPOCHS=15 uv run python -m src.token_level_distillation
EPOCHS=15 uv run python -m src.cross_tokenizer
EPOCHS=15 uv run python -m src.on_policy
uv run python -m src.eval_gsm8k --n 100 --adapters base distilled_sequence_level \
  distilled_token_level distilled_student_onpolicy distilled_student_uld
```

Run sequentially on this machine — parallel launches hit an NVML race in
`caching_allocator_warmup` (driver/library mismatch in this env) and crash
during model loading. Total wall time: ~2h7m, of which 114 min is on-policy
alone.
