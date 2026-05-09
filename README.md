# Distillation - how does it work?

A runnable project that demonstrates **LLM knowledge distillation**: transferring the capability of a large "teacher" model into a much smaller "student" model.

I built this to see what distillation is. It is easy to spend a lot of time reading about it and work with abstractions in your mind. It is much better to see and taste what distillation is.

## Four distillation methods in this repo

The teacher generates step-by-step solutions to GSM8K math problems. The student (Qwen2.5-0.5B) is then trained four different ways:

1. **Sequence-level** 
2. **Token-level / logit KL**
3. **On-policy / GKD**
4. **Cross-tokenizer / ULD**

[In progress]
