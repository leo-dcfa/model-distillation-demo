import json
from pathlib import Path
from typing import Final

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.config import EPOCHS, STUDENT_MODEL, TEACHER_DATA
from src.logger import JsonlTrainerCallback


OUTPUT_DIR: Final[str] = "./distilled_student"
RUN_NAME: Final[str] = "sequence_level"


def load_dataset_from_jsonl(path: str, tokenizer: TokenizersBackend) -> Dataset:
    rows = []
    with Path.open(Path(path)) as f:
        for line in f:
            ex = json.loads(line)
            messages = [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["teacher_solution"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            rows.append({"text": text})
    return Dataset.from_list(rows)


def main():
    print(f"Loading student: {STUDENT_MODEL}")
    tokenizer = get_tokenizer(STUDENT_MODEL)
    print(f"Loaded tokenizer: {tokenizer}")
    model = get_model(STUDENT_MODEL)
    print(f"Loaded model: {model}")
    print(f"Model device: {model.device}")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading teacher data from {TEACHER_DATA}")
    train_ds = load_dataset_from_jsonl(TEACHER_DATA, tokenizer=tokenizer)
    print(f"Training examples: {len(train_ds)}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=1024,
        packing=False,
        dataset_text_field="text",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[JsonlTrainerCallback(run_name=RUN_NAME)],
    )

    print("\nStarting distillation training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nSaved distilled student adapter to {OUTPUT_DIR}")


def get_tokenizer(model_name: str) -> TokenizersBackend:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not isinstance(tokenizer, TokenizersBackend):
        raise TypeError(f"Cannot proceed; expected Tonenizer backend, got {type(tokenizer)}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )


if __name__ == "__main__":
    main()
