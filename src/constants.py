from typing import Final

import torch


DEFAULT_STUDENT: Final[str] = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATA: Final[str] = "teacher_data.jsonl"
DEFAULT_EPOCHS: Final[int] = 2
DEFAULT_OUTPUT_DIR: Final[str] = "./distilled_student"
DEFAULT_TEACHER: Final[str] = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_N: Final[int] = 2000
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
