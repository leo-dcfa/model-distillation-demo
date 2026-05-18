import json
from pathlib import Path

from datasets.arrow_dataset import Dataset
from transformers import TokenizersBackend


TEMPERATURE = 2.0
MAX_LENGTH = 1024
EPOCHS = 3
BATCH_SIZE = 2  # small batches since we have two models loaded in memory
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.03


class TeacherDataSet(Dataset):
    def __init__(self, path: str, tokenizer: TokenizersBackend, max_length=MAX_LENGTH):
        self.examples = []
        with Path.open(Path(path)) as f:
            for line in f:
                ex = json.loads(line)
                prompt_msgs = [{"role": "user", "content": ex["requestion"]}]
                full_msgs = [*prompt_msgs, {"role": "assistant", "content": ex["teacher_solution"]}]
                #  the user's question only, with a "now it's the assistant's turn" marker appended (add_generation_prompt=True)
                prompt_ids = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
                #  the user's question and the teacher's full response
                full_ids = tokenizer.apply_chat_template(full_msgs)
                # input ids is the full tokenized conversation (user prompt + assistant solution)
                # prompt_len is the length of the prompt portion only
                self.examples.append(
                    {
                        "input_ids": full_ids,
                        "prompt_len": len(prompt_ids),
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        return self.examples[key]


def collate(batch, pad_token_id):
    pass


def kl_loss():
    pass


def ce_loss():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
