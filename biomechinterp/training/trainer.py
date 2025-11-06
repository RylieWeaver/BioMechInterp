# General
import json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union

# Torch
import torch

# biomechinterp
from biomechinterp.models import SparseAutoencoder
from biomechinterp.data import move_to
from biomechinterp.utils import Config, resolve_device
from .loss import sparse_autoencoder_loss
from .optimizer import OptHandler
from .loader import LoaderHandler
from .logger import Logger



class SAETrainerConfig(Config):
    def __init__(
            self,
            last_epoch: int = -1,
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
        self.last_epoch = last_epoch
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
    
    @staticmethod
    def load(path: Path) -> "SAETrainerConfig":
        """
        The keys here for OptHandler, LoaderHandler, and Logger 
        must match the varnames of their respective instances.
        """
        with path.open("r") as f:
            cfg = json.load(f)
        return SAETrainerConfig(
            last_epoch=cfg["last_epoch"],
            epochs=cfg["epochs"],
            l1_coefficient=cfg["l1_coefficient"],
            gradient_clip=cfg["gradient_clip"],
            device=cfg.get("device", "auto"),
            checkpoint_dir=cfg.get("checkpoint_dir", None),
            save_every=cfg.get("save_every", None),
            opt_handler=OptHandler(**cfg["opt_handler"]),
            loader_handler=LoaderHandler(**cfg["loader_handler"]),
            logger=Logger(**cfg["logger"]),
        )


class SAETrainer:
    def __init__(self, config, model) -> None:
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
        self.optimizer = self.opt_handler.build(self.model.parameters())
    
    def set_loaders(self, train_ds, val_ds, test_ds):
        self.train_loader, self.val_loader, self.test_loader = self.loader_handler.build(train_ds, val_ds, test_ds)

    def _run_batch(self, activations):
        activations = move_to(activations, self.device)
        reconstruction, sparse_rep = self.model(activations)
        return reconstruction, sparse_rep

    def _compute_loss(self, reconstruction, activations, sparse_rep):
        loss = sparse_autoencoder_loss(
            activations,
            reconstruction,
            sparse_rep,
            l1_coefficient=self.cfg.l1_coefficient,
        )
        return loss
    
    def _loop_without_grad(self, loader, desc):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                reconstruction, sparse_rep = self._run_batch(batch)
                loss = self._compute_loss(reconstruction, batch, sparse_rep)
                total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        return epoch_loss

    def _loop_with_grad(self, loader, desc):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=desc, leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            reconstruction, sparse_rep = self._run_batch(batch)
            loss = self._compute_loss(reconstruction, batch, sparse_rep)
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

    def train(self, last_epoch=None, epochs=None) -> None:
        # Setup
        self.logger.log_time("Train Start")
        self.cfg.last_epoch = last_epoch if last_epoch is not None else self.cfg.last_epoch
        epochs = epochs if epochs is not None else self.cfg.epochs

        # Initial evaluation before training
        if self.cfg.last_epoch == -1:
            curr_epoch = self.cfg.last_epoch + 1
            train_loss = self._loop_without_grad(self.train_loader, desc="Train w/o Grad")
            val_loss = self._loop_without_grad(self.val_loader, desc="Val w/o Grad")
            test_loss = self._loop_without_grad(self.test_loader, desc="Test w/o Grad")
            self.logger.log(f"[Epoch {curr_epoch}]: Train w/o Grad {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}")
            self.cfg.last_epoch += 1

        # Training loop
        while self.cfg.last_epoch < epochs:
            curr_epoch = self.cfg.last_epoch + 1
            self._run_epoch(curr_epoch)
            self.cfg.last_epoch = curr_epoch
            if self.cfg.checkpoint_dir and self.cfg.save_every and curr_epoch % self.cfg.save_every == 0:
                self._save_checkpoint(curr_epoch)

        # Cleanup
        self.logger.log_time("Train End")

    def _save_checkpoint(self, epoch: int):
        save_dir = self.cfg.checkpoint_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        self.model.cfg.save(save_dir / "model_config.json")
        self.cfg.save(save_dir / "trainer_config.json")
        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")

    @staticmethod
    def load(dir: Path) -> "SAETrainer":
        # Load model
        model = SparseAutoencoder.load(dir)

        # Load trainer
        trainer_cfg_path = dir / "trainer_config.json"
        with trainer_cfg_path.open("r") as f:
            cfg = json.load(f)
        trainer_cfg = SAETrainerConfig(
            last_epoch=cfg.get("last_epoch", -1),
            epochs=cfg.get("epochs", 20),
            l1_coefficient=cfg.get("l1_coefficient", 1e-3),
            gradient_clip=cfg.get("gradient_clip"),
            device=cfg.get("device", "auto"),
            checkpoint_dir=cfg.get("checkpoint_dir"),
            save_every=cfg.get("save_every"),
            opt_handler=OptHandler(**cfg.get("opt_handler", {})),
            loader_handler=LoaderHandler(**cfg.get("loader_handler", {})),
            logger=Logger(**cfg.get("logger", {})),
        )
        trainer = SAETrainer(trainer_cfg, model)

        # Load optimizer state
        opt_state = torch.load(dir / "optimizer.pt")
        trainer.optimizer.load_state_dict(opt_state)
        return trainer
