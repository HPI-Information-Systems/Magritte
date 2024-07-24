import sys, os
import torch
import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader
from .table_dataset import TableDataset
from transformers import BertTokenizer


class ColumnTypeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_datapath: str,
        val_datapath: str = "",
        test_datapath: str = "",
        cv_fold: int = 0,
        labels_path: str = None,
        n_files: int = None,
        batch_size: int = 8,
        num_workers: int = 32,
        shuffle: bool = True,
        save_path: str = None,
        dataset: str = "all",
    ):
        super().__init__()
        self.train_datapath = train_datapath
        self.val_datapath = val_datapath
        self.test_datapath = test_datapath

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.n_files = n_files
        self.dataset = dataset

        self.data_train = None
        self.data_val = None
        self.save_path = save_path
        self.cv_fold = cv_fold

        with open(labels_path, "r") as label_file:
            labels = label_file.readlines()
        self.label_dict = {label.strip(): i for i, label in enumerate(labels)}

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def prepare_data(self):
        """Give paths to jsonl files, create the corresponding tokenized versions if they do not exist"""

        self.train_dataset = TableDataset(
            self.train_datapath,
            save_path=self.save_path + f"/{self.dataset}_train_{self.cv_fold}.pt",
            tokenizer=self.tokenizer,
            label_dict=self.label_dict,
            n_files=self.n_files,
        )

        if self.val_datapath != "":
            self.val_dataset = TableDataset(
                self.val_datapath,
                save_path=self.save_path + f"/{self.dataset}_val_{self.cv_fold}.pt",
                tokenizer=self.tokenizer,
                label_dict=self.label_dict,
                n_files=self.n_files,
            )

        if self.test_datapath != "":
            self.test_dataset = TableDataset(
                self.test_datapath,
                save_path=self.save_path + f"/{self.dataset}_test.pt",
                tokenizer=self.tokenizer,
                label_dict=self.label_dict,
                n_files=self.n_files,
            )

    def setup(self, stage=None):
        pass
        # raise ValueError("Stage not recognized: ", stage)

    def _common_dataloader(self, dataset, shuffle=True):

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self):
        return self._common_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._common_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._common_dataloader(self.test_dataset, shuffle=False)
