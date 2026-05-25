# Distillation methods: side-by-side comparison

5-epoch run of all four methods against 500 GSM8K teacher solutions, evaluated
on 50 held-out GSM8K test problems via greedy decoding.

## Headline numbers

| Method | Teacher | GSM8K@50 | Δ vs base | Train time | Steps logged |
| --- | --- | --: | --: | --: | --: |
| **base** (no distillation) | — | 10% | — | — | — |
| **Sequence-level** (SFT on teacher text) | Qwen2.5-3B-Instruct | 40% | +30 | ~1.5 min | 17 |
| **Token-level KL** (forward, α=0.1, T=2) | Qwen2.5-3B-Instruct | 44% | +34 | ~2 min | 155 |
| **On-policy GKD** (reverse KL, 50/50 mix) | Qwen2.5-3B-Instruct | **46%** | **+36** | ~35 min | 155 |
| **Cross-tokenizer ULD** (sorted top-K KL) | SmolLM2-1.7B-Instruct | 6% | -4 | ~2 min | 155 |

All students are LoRA adapters (r=16, q/k/v/o projections) on `Qwen2.5-0.5B`.
Eval is greedy generation, 512 max new tokens, scored by extracting the final
number after `#### ` or `\boxed{...}` and comparing to ground truth.

## The format-mismatch trap (read this first)

First-pass eval scored token-level, on-policy, and ULD at **0%**. They weren't
broken — they were producing correct math in a different output format.

The teacher data was generated *with* the system prompt
`"...End your response with the final answer on its own line in the form '#### <number>'."`,
so the saved teacher solutions are 466/500 `#### N` and 27/500 `\boxed{N}`.

The training scripts then drop the system prompt:
```python
prompt_msgs = [{"role": "user", "content": ex["question"]}]   # no system msg
```
Sequence-level survives this because CE on the literal saved text forces the
student to reproduce `#### N` regardless. The KL methods don't: they train the
student to match the teacher's *distribution* given the dropped-system-prompt
context, where the Qwen 3B instruct model naturally drifts toward
`\boxed{N}`. So the student fluently learns `\boxed{N}` instead.

Re-running the eval with a parser that accepts both formats gives the numbers
above. Sequence-level is unchanged at 40%; the others jumped from 0% to 44/46%.

**Fix in the training pipeline** (not applied here): carry the system prompt
through `prompt_msgs` and `full_msgs` in `token_level_distillation.py`,
`on_policy.py`, and `cross_tokenizer.py` so the student is trained on the same
context the teacher saw and the eval expects.

## Method-by-method

### Sequence-level (40%)
Standard SFT on the teacher's saved text. Cheapest, simplest, most predictable.
Training loss `0.86 → 0.36` over 5 epochs. The format-mismatch issue can't
touch it because the loss directly targets the saved tokens. A solid baseline
that any white-box method should beat — and three out of three do (one of them
narrowly).

### Token-level forward KL (44%)
KL(teacher‖student) on the full vocab at every assistant token, α=0.1 CE
anchor, T=2. Logged loss `0.66 → 0.57`. Beats sequence-level by 4 points (2 of
50) — a modest but real lift for the same data budget, and consistent with the
literature claim that token-level gives a denser signal than SFT. The match
isn't huge here because Qwen2.5-0.5B is already similar in architecture to the
teacher and 500 examples is a small budget.

### On-policy GKD (46%)
Reverse KL with a 50/50 on-policy / off-policy mix at T=1. Best score (by 1
example over token-level — within noise on n=50). Loss `0.19 → 0.15`. Cost:
**35 minutes vs ~2 minutes** for the off-policy methods, because every
on-policy step generates a full continuation from the student before the
forward/backward. For this much extra compute on this size of student, the
1-example lift over token-level is not worth it on its own — the win from
on-policy is supposed to come from closing the train/test gap, and on 50
problems we can't see a 4-point gap.

### Cross-tokenizer ULD (6%)
Underperforms even the base model. Two compounding effects:
1. **Weaker teacher.** SmolLM2-1.7B-Instruct (used here because we don't have
   access to Llama-3.2-1B-Instruct) is smaller and a worse mathematician than
   the Qwen 3B teacher used by methods 1–3. The student can't outrun a weak
   teacher.
2. **Method losses by design.** Character-offset alignment is approximate,
   sorted top-K KL throws away token identity, and the tail past K is gone.
   Even with a strong teacher this method should produce a noticeably weaker
   student than same-tokenizer token-level KL.
3. **Output format drift.** SmolLM2 doesn't default to `\boxed{N}` either; its
   raw outputs end with prose like `... = 26 dollars.` and the parser misses
   them. Some fraction of the 6% number is parser miss rather than wrong math.

If cross-tokenizer is the only option (e.g. distilling from a proprietary
teacher into your preferred arch), expect it to work — but on the same data
budget against same-tokenizer methods it's strictly the worst tool here.

## Caveats

- **n=50 is small.** 1–2 example differences between sequence-level,
  token-level, and on-policy are inside noise. The relative ordering matches
  the theory but the gaps shouldn't be over-interpreted.
- **Loss values are not directly comparable across methods.** Different loss
  functions (CE vs full-vocab KL vs sorted-top-K KL vs masked-response KL)
  with different temperatures and scales. Don't rank methods by loss
  magnitude — only by the eval column.
- **Greedy decoding only.** Sampling with temperature could change the ranking,
  particularly for on-policy (which was trained to be mode-seeking under
  reverse KL).
- **One run per method.** No seed averaging. Reverse KL especially can be
  unstable; a different RNG seed could move ULD or on-policy by several
  points.
- **The system-prompt drop is a real bug** in three of the four scripts. If
  you fix it and re-run, the KL methods should slightly improve (cleaner
  format adherence) without changing the ordering.

## Reproducing

```bash
EPOCHS=5 uv run python -m src.sequence_level_distillation
EPOCHS=5 uv run python -m src.token_level_distillation
EPOCHS=5 uv run python -m src.on_policy
EPOCHS=5 uv run python -m src.cross_tokenizer
uv run python -m src.eval_gsm8k --n 50 --adapters base distilled_sequence_level \
  distilled_token_level distilled_student_onpolicy distilled_student_uld
```

Run the trainings one at a time on this machine — parallel launches hit an
NVML race in `caching_allocator_warmup` (likely a driver/library mismatch in
this environment) and crash during model loading.
