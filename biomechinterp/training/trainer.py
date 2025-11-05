# General
import json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union

# Torch
import torch

# biomechinterp
from biomechinterp.data import move_to
from biomechinterp.utils import resolve_device
from loss import sparse_autoencoder_loss
from optimizer import OptHandler
from loader import LoaderHandler
from logger import Logger
from utils import Config



class SAETrainerConfig(Config):
    def __init__(
            self,
            epochs: int = 20,
            l1_coefficient: float = 1e-3,
            gradient_clip: Optional[float] = None,
            opt_handler=None,
            loader_handler=None,
            logger=None,
            checkpoint_dir: Optional[Union[Path, str]] = None,
            save_every: Optional[int] = None,
            device: str = "auto"
    ) -> None:
        self.epochs = epochs
        self.l1_coefficient = l1_coefficient
        self.gradient_clip = gradient_clip
        self.opt_handler = opt_handler if opt_handler is not None else OptHandler()
        self.loader_handler = loader_handler if loader_handler is not None else LoaderHandler()
        self.logger = logger if logger is not None else Logger()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.device = device
    
    def save(self, path: Path):
        cfg = self.to_dict()
        with path.open("w") as f:
            json.dump(cfg, f, indent=4)

    @staticmethod
    def load(path: Path):
        """
        The keys here for OptHandler, LoaderHandler, and Logger 
        must match the varnames of their respective instances.
        """
        with path.open("r") as f:
            cfg = json.load(f)
        return SAETrainerConfig(
            epochs=cfg["epochs"],
            l1_coefficient=cfg["l1_coefficient"],
            gradient_clip=cfg.get("gradient_clip"),
            device=cfg.get("device", "auto"),
            checkpoint_dir=cfg.get("checkpoint_dir"),
            save_every=cfg.get("save_every"),
            opt_handler=OptHandler(**cfg["opt_handler"]),
            loader_handler=LoaderHandler(**cfg["loader_handler"]),
            logger=Logger(**cfg["logger"]),
        )


class SAETrainer:
    def __init__(self, config, model, train_ds, val_ds, test_ds) -> None:
        # Read args
        self.cfg = config
        self.device = resolve_device(self.cfg.device)
        self.model = model.to(self.device)
        self.checkpoint_dir = self.cfg.checkpoint_dir
        self.save_every = self.cfg.save_every
        self.loader_handler = self.cfg.loader_handler
        self.opt_handler = self.cfg.opt_handler
        self.logger = self.cfg.logger

        # Init objects
        self.train_loader, self.val_loader, self.test_loader = self.loader_handler.build(train_ds, val_ds, test_ds)
        self.optimizer = self.opt_handler.build(self.model.parameters())

    def _run_batch(self, batch):
        batch = move_to(batch, self.device)
        preds = self.model(batch["inputs"])
        labels = batch["labels"]
        return preds, labels
    
    def _compute_loss(self, preds, labels):
        loss = sparse_autoencoder_loss(
            preds,
            labels,
            l1_coefficient=self.cfg.l1_coefficient,
        )
        return loss
    
    def _loop_without_grad(self, loader, desc):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                preds, labels = self._run_batch(batch)
                loss = self._compute_loss(preds, labels)
                total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        return epoch_loss

    def _loop_with_grad(self, loader, desc):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=desc, leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            preds, labels = self._run_batch(batch)
            loss = self._compute_loss(preds, labels)
            loss.backward()
            total_loss += loss.item()
            if self.cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
            self.optimizer.step()
        
        epoch_loss = total_loss / len(loader)
        return epoch_loss
    
    def _run_epoch(self, epoch: int):
        train_loss_grad = self._loop_with_grad(self.train_loader, desc=f"Train w/ Grad")
        train_loss_no_grad = self._loop_without_grad(self.train_loader, desc="Train w/o Grad")
        val_loss = self._loop_without_grad(self.val_loader, desc="Val w/o Grad")
        test_loss = self._loop_without_grad(self.test_loader, desc="Test w/o Grad")
        self.logger.log(f"[Epoch {epoch}]: Train w/ Grad: {train_loss_grad:.6f}, Train w/o Grad: {train_loss_no_grad:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}")

    def train(self) -> None:
        # Setup
        self.logger.log_time("Train Start")

        # Initial evaluation before training
        train_loss = self._loop_without_grad(self.train_loader, desc="Train w/o Grad")
        val_loss = self._loop_without_grad(self.val_loader, desc="Val w/o Grad")
        test_loss = self._loop_without_grad(self.test_loader, desc="Test w/o Grad")
        self.logger.log(f"[Epoch 0]: Train w/o Grad {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}")

        # Training loop
        for epoch in range(1, self.cfg.epochs + 1):
            self._run_epoch(epoch)
            if self.cfg.checkpoint_dir and self.cfg.save_every and epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch)

        # Cleanup
        self.logger.log_time("Train End")

    def _save_checkpoint(self, epoch: int):
        save_dir = self.cfg.checkpoint_dir / f"epoch_{epoch}.pt"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")
        self.cfg.save(save_dir / "config.json")
