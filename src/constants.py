from typing import Final


DEFAULT_STUDENT: Final[str] = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATA: Final[str] = "teacher_data.jsonl"
DEFAULT_EPOCHS: Final[int] = 2
DEFAULT_OUTPUT_DIR: Final[str] = "./distilled_student"
DEFAULT_TEACHER: Final[str] = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_NUMBER_EXAMPLES: Final[int] = 2000
