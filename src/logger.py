import json
import time
from pathlib import Path

from transformers import TrainerCallback


class MetricsLogger:
    def __init__(self, run_name, log_dir="logs"):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.path = log_path / f"{run_name}.jsonl"
        self.run_name = run_name
        with self.path.open("w"):
            pass
        print(f"[MetricsLogger] Logging '{run_name}' to {self.path}")

    def log(self, **kwargs):
        record = {"timestamp": time.time(), **kwargs}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


class JsonlTrainerCallback(TrainerCallback):
    def __init__(self, run_name, log_dir="logs"):
        self.logger = MetricsLogger(run_name, log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        payload = {"step": state.global_step, "epoch": state.epoch}
        payload.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        self.logger.log(**payload)
