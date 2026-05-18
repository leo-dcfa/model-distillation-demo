# Distillation: how does it work?

A runnable project that demonstrates LLM knowledge distillation: transferring the capability of a large "teacher" model into a much smaller "student" model.

I built this to see better understand what distillation is. It is easy to spend a lot of time reading about it and work with abstractions in your mind. It is much better to see and taste what distillation is.

I believe learning sediments better if I do it myself. Hence, this is a project coded by hand; Claude Opus 4.7 and Qwen 3.6 35b3a were my guides.

## Four distillation methods in this repo

The teacher generates step-by-step solutions to [GSM8K math problems](https://huggingface.co/datasets/openai/gsm8k). The student (Qwen2.5-0.5B) is then trained four different ways:

1. Sequence-level
2. Token-level / logit KL
3. On-policy / GKD
4. Cross-tokenizer / ULD

## Method 1: Sequence-level distillation

The simplest approach. The teacher generates a completion for each prompt; the
student is fine-tuned with standard next-token cross-entropy on those completions.
We're treating the teacher's output text as ground truth and doing ordinary SFT.

**Pros.** Works against any API (you only need to call generate). Cheap to
implement. No tokenizer constraints — the student can have a completely
different vocabulary from the teacher. This is how DeepSeek's R1-Distill
models were trained.

**Cons.** Throws away everything except the teacher's argmax at each token.
If the teacher was 60% sure about token X and 38% sure about a near-synonym Y,
the student is taught that Y is wrong. A lot of useful uncertainty signal is lost.

## Method 2: Token-level (logit) distillation

The direct LLM analog of Hinton's classical method. At every position in a
sequence, compute the KL divergence between the teacher's full next-token
distribution (over the whole vocabulary) and the student's. Loss is averaged
across positions where the assistant is talking (we mask out user prompt tokens).

**Pros.** The student receives a much denser training signal — instead of "the
right token is X," it learns "given this context, here's the full shape of
plausible continuations." Typically yields a noticeably stronger student than
sequence-level for the same data budget.

**Cons.** Requires white-box access to the teacher *and* matching tokenizers.
You can't do logit distillation from GPT-4 to Llama, for example — different
vocabularies, so the distributions live in non-comparable spaces. Memory cost
is also higher because you're keeping the teacher loaded and running forward
passes through it on every batch.

### Forward vs. reverse KL

`token_level.py` accepts a `--kl` flag with two options:

- **`--kl forward`** (default): minimizes `KL(teacher || student)`. This is
  *mode-covering* — the student is penalized for putting low probability where
  the teacher puts high probability. It tries to cover all of the teacher's
  behaviors, including unlikely ones. Can lead to a hedging, hallucination-prone
  student because it spreads its probability mass.

- **`--kl reverse`**: minimizes `KL(student || teacher)`. This is *mode-seeking*
  — the student is penalized only when *it* puts mass where the teacher doesn't.
  Result: the student concentrates on a high-probability subset of the teacher's
  behavior. The MiniLLM paper (Gu et al. 2024) showed this produces more focused,
  less hallucinatory students.

Most modern LLM distillation uses reverse KL or a hybrid. Our on-policy script
uses reverse KL by default, following GKD.
