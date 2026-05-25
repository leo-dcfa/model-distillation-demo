from typing import Final


DEFAULT_STUDENT: Final[str] = "Qwen/Qwen2.5-0.5B"
DEFAULT_EPOCHS: Final[int] = 2
DEFAULT_TEACHER: Final[str] = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_NUMBER_EXAMPLES: Final[int] = 500

# Used both when generating teacher solutions and when training the student.
# Must match across the two — otherwise the student is trained on contexts the
# teacher never saw, and at inference falls back to whatever format the teacher
# defaults to without a system prompt (\boxed{N} for Qwen-Instruct, prose for
# SmolLM2, etc.).
SYSTEM_PROMPT: Final[str] = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your reasoning clearly. End your response with the final "
    "answer on its own line in the form '#### <number>'."
)
