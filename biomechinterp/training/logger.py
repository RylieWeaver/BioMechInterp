# General
import time
from pathlib import Path

# Torch

# biomechinterp
from biomechinterp.utils import Config



class Logger(Config):
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_time(self, message: str = "") -> None:
        print(message)
        log_path = self.log_dir / "timestamp.log"
        with log_path.open("a") as fh:
            fh.write(f"{message} Timestamp: {time.time()}\n")

    def log(self, message: str) -> None:
        print(message)
        log_path = self.log_dir / "training.log"
        with log_path.open("a") as fh:
            fh.write(message + "\n")
