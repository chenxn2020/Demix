"""Base DataModule class."""
from pathlib import Path
from typing import Dict
import argparse
import os
from torch.utils.data import DataLoader
from .base_data_module import *
import pytorch_lightning as pl


class KGDataModule(BaseDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(
        self, args: argparse.Namespace = None, train_sampler=None, test_sampler=None
    ) -> None:
        super().__init__(args)
        self.eval_bs = self.args.eval_bs
        self.num_workers = self.args.num_workers
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler


    def get_data_config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "num_training_steps": self.num_training_steps,
            "num_labels": self.num_labels,
        }

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.data_train = self.train_sampler.get_train()
        # self.data_train = self.train_sampler.get_train_caches()
        self.data_val   = self.train_sampler.get_valid()
        self.data_test  = self.train_sampler.get_test()

    def get_train_bs(self):
        if self.args.num_batches != 0:
            self.args.train_bs = len(self.data_train) // self.args.num_batches
        elif self.args.train_bs == 0:
            raise ValueError("train_bs or num_batches must specify one")
        return self.args.train_bs

    def train_dataloader(self):
        self.train_bs = self.get_train_bs()
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.train_bs,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_sampler.sampling,
            # collate_fn=self.train_sampler.sampling_cache,
            # collate_fn=self.train_sampler.batch_sampling,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.eval_bs,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_sampler.sampling,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.eval_bs,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_sampler.sampling,
        )