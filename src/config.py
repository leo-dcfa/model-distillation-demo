import os


EPOCHS: int = int(os.environ.get("EPOCHS", "2"))
STUDENT_MODEL: str = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B")
TEACHER_DATA = "teacher_data.jsonl"
