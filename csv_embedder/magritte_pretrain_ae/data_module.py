import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from .dataset import CsvFileDataset
from csv_embedder.pattern_tokenizer import PatternTokenizer


class CsvFileDataModule(pl.LightningDataModule):

    def __init__(self, data_path: str,
                 token_vocab: Vocab,
                 tokenizer: PatternTokenizer,
                 n_files: int,
                 max_rows: int,
                 max_len: int,
                 batch_size: int,
                 num_workers: int,
                 save_dir: str,
                 train_val_split: float = 0.8,
                 shuffle: bool = True,):

        super().__init__()
        self.data_path = data_path
        self.token_vocab = token_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.n_files = n_files
        self.tokenizer = tokenizer
        self.train_files = None
        self.val_files = None
        self.shuffle = shuffle

        self.train_val_split = train_val_split
        self.save_dir = save_dir

        SEED = 42
        self.rng = np.random.default_rng(SEED)

    def prepare_data(self):

        all_files = os.listdir(self.data_path)

        all_files = os.listdir(self.data_path)
        n_f = min(self.n_files, len(all_files))
        n_train_files = int(n_f * self.train_val_split)

        files = self.rng.choice(all_files, n_f, replace=False)
        self.train_files = files[:n_train_files]
        self.val_files = files[int(self.train_val_split * len(files)):]
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.save_dir}/train_files.txt", "w") as f:
            for file in self.train_files:
                f.write(f"{file}\n")
        with open(f"{self.save_dir}/val_files.txt", "w") as f:
            for file in self.val_files:
                f.write(f"{file}\n")

    def setup(self, stage=None):
        pass

    def _common_dataloader(self, csv_files, shuffle=True):
        dataset = CsvFileDataset(csv_files,
                                            data_path=self.data_path,
                                            token_vocab=self.token_vocab,
                                            max_rows=self.max_rows,
                                            max_len=self.max_len,
                                            tokenizer=self.tokenizer, )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._common_dataloader(self.train_files, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._common_dataloader(self.val_files, shuffle=False)