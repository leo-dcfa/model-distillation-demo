import os
from typing import Final

from src.constants import DEFAULT_EPOCHS, DEFAULT_STUDENT, DEFAULT_TEACHER


EPOCHS: int = int(os.environ.get("EPOCHS", DEFAULT_EPOCHS))
STUDENT_MODEL: Final[str] = os.environ.get("STUDENT_MODEL", DEFAULT_STUDENT)
TEACHER_MODE: Final[str] = os.environ.get("TEACHER_MODEL", DEFAULT_TEACHER)
TEACHER_DATA: Final[str] = "teacher_data.jsonl"
