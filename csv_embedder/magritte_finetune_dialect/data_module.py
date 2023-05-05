import csv
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from csv_embedder.magritte_finetune_dialect.dataset import DialectDataset
from csv_embedder.pattern_tokenizer import PatternTokenizer


class DialectDataModule(pl.LightningDataModule):

    def __init__(self, train_data_path: str,
                 val_data_path: str,
                 token_vocab: Vocab,
                 label_vocab: Vocab,
                 n_files: int,
                 tokenizer: PatternTokenizer,
                 max_rows: int,
                 max_len: int,
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool = True, ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.n_files = n_files
        self.tokenizer = tokenizer
        self.data_train = None
        self.data_val = None

    def prepare_data(self):
        for dataset_path in [self.train_data_path, self.val_data_path]:
            dialect_file = f"{dataset_path}/dialect_annotations.csv"
            if not os.path.isfile(dialect_file):
                print(f"Annotation file not found in {dialect_file}, creating it...")
                dialect_path = f"{dataset_path}/dialect/"

                dialect_list = []
                if not os.path.isdir(dialect_path):
                    os.mkdir(dialect_path)
                for file in os.listdir(dialect_path):
                    if file.endswith(".json"):
                        with open(f"{dialect_path}/{file}") as f:
                            annotation = json.load(f)
                        dialect_list.append({"filename": annotation["filename"], **annotation["dialect"]})
                        if annotation["filename"] not in os.listdir(dataset_path + "/csv/"):
                            print("File", annotation["filename"]," not found in csv dir")
                            raise AssertionError

                df = pd.DataFrame(dialect_list)
                Path(os.path.dirname(dialect_file)).mkdir(parents=True, exist_ok=True)
                df.to_csv(dialect_file, index=False, quoting=csv.QUOTE_ALL)

    def setup(self, stage=None):
        pass

    def _common_dataloader(self, data_path, shuffle=True):
        dialect_file = f"{data_path}/dialect_annotations.csv"
        annotations_df = pd.read_csv(dialect_file)
        annotations_df = annotations_df.fillna("")

        dataset = DialectDataset(annotations_df,
                                            data_path=data_path,
                                            token_vocab=self.token_vocab,
                                            label_vocab=self.label_vocab,
                                            max_rows=self.max_rows,
                                            max_len=self.max_len,
                                            tokenizer=self.tokenizer,
                                            n_files=self.n_files, )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        data_path = self.train_data_path
        return self._common_dataloader(data_path, shuffle=self.shuffle)

    def val_dataloader(self):
        data_path = self.val_data_path
        return self._common_dataloader(data_path, shuffle=False)

    def test_dataloader(self,dataset):
        return self._common_dataloader(dataset)
