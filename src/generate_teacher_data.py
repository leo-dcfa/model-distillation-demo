from src.config import TEACHER_MODEL
from src.utils import get_model


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your reasoning clearly. End your response with the final "
    "answer on its own line in the form '#### <number>'."
)


def main():
    print(f"Loading teacher {TEACHER_MODEL}")

    model = get_model(TEACHER_MODEL)
    # here we say `not training, we are doing inference`
    model.eval()
