# General
from dataclasses import dataclass
from typing import Tuple

# Torch
from torch.utils.data import DataLoader

# biomechinterp
from biomechinterp.utils import Config



@dataclass
class LoaderHandler(Config):
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    
    def build(self, train_ds, val_ds, test_ds, collate_fn=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader, test_loader
