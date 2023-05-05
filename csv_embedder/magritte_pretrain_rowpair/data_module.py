import csv
import json
import pdb
import os
from pathlib import Path

import chardet
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from numpy.random._generator import default_rng
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import Vocab

from .dataset import RowPairDataset
from csv_embedder.pattern_tokenizer import PatternTokenizer

def contains_special(token):
    x = str(token)
    return ("d" not in x) and ("L" not in x) and ("l" not in x) and ("A" not in x) and ("S" not in x) and ("T" not in x)


class RowPairDataModule(pl.LightningDataModule):

    def __init__(self, train_data_path: str,
                 val_data_path: str,
                 token_vocab: Vocab,
                 n_pairs: int,
                 tokenizer: PatternTokenizer,
                 max_rows: int,
                 max_len: int,
                 batch_size: int,
                 num_workers: int,
                 max_percent = 0.15,
                 mask_special_only = True,
                 positive_ratio = 0.5,
                 tmp_path = "results/tmp/",
                 shuffle = True,
                 seed = 42,
                 pad_token = "[PAD]",
                 cls_token = "[CLS]",
                 sep_token = "[SEP]",
                 mask_token = "[MASK]", ):

        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.token_vocab = token_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.n_pairs = n_pairs
        self.tokenizer = tokenizer
        self.data_train = None
        self.data_val = None
        self.max_percent = max_percent
        self.mask_special_only = mask_special_only
        self.positive_ratio = positive_ratio
        self.tmp_path = tmp_path
        self.shuffle = shuffle

        self.train_instance_file = f"{self.train_data_path}/instances_max_len{self.max_len}_{'mask_special' if self.mask_special_only else 'mask_all'}.csv"
        self.val_instance_file = f"{self.val_data_path}/instances_max_len{self.max_len}_{'mask_special' if self.mask_special_only else 'mask_all'}.csv"
        self.max_masks = int(np.ceil(self.max_percent * max_len))

        self._seed = seed
        self.rng = default_rng(self._seed)

        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token



    def prepare_data(self):
        if not os.path.isfile(self.train_instance_file):
                print(f"Training instances not found in {self.train_instance_file}, creating them now...")
                self.materialize_instances(self.train_instance_file)

        if not os.path.isfile(self.val_instance_file):
                print(f"Val instances not found in {self.val_instance_file}, creating them now...")
                self.materialize_instances(self.val_instance_file)

    def setup(self, stage=None):
        if stage == 'fit':
            instance_path = self.train_instance_file
            data_path = self.train_data_path
        elif stage == 'val':
            instance_path = self.val_instance_file
            data_path = self.val_data_path
        else:
            return None

        instance_df = pd.read_csv(instance_path, nrows=self.n_pairs, keep_default_na=False)

        n_positive = len(instance_df[instance_df["same_file"] == True])
        n_negative = len(instance_df[instance_df["same_file"] == False])
        print(f"Length of instances read: {len(instance_df)}")
        print(f"Class balance of pairs read: {n_positive/len(instance_df)} positive, {n_negative/len(instance_df)} negative")

        self.dataset = RowPairDataset(instance_df,
                                    data_path=data_path,
                                    token_vocab=self.token_vocab,
                                    max_rows=self.max_rows,
                                    max_len=self.max_len,
                                    tokenizer=self.tokenizer,
                                    max_percent = self.max_percent,
                                    n_pairs=self.n_pairs, )

    def _common_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._common_dataloader(self.dataset, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._common_dataloader(self.dataset, shuffle=False)

    def materialize_instances(self, dataset_path):
        dataset_name, split_name = dataset_path.split("/")[1:]

        csv_path = dataset_path.split("/")[0] + f"/{dataset_name}/csv/"
        csv_files = sorted([f for f in os.listdir(csv_path)])
        n_files = len(csv_files)

        if split_name in ("predev", "test"):
            n_pairs = int(min(10e6, self.pair_count * 0.1))
        else:
            n_pairs = self.pair_count

        n_positive = int(n_pairs * self.positive_ratio)
        n_negative = n_pairs - n_positive
        positive_files = np.asarray(sorted(self.rng.choice(range(n_files), size=(n_positive), replace=True)))

        negative_files1 = self.rng.choice(range(n_files), size=(n_negative), replace=True)
        negative_files2 = self.rng.choice(range(n_files), size=(n_negative), replace=True)
        for idx, x in enumerate(negative_files1==negative_files2):
            if x:
                new_lst = list(range(0,negative_files2[idx]))+list(range(negative_files2[idx]+1,n_files))
                negative_files2[idx] = self.rng.choice(new_lst, size=1, replace=False)

        negative_files1 = np.asarray(sorted(negative_files1))
        negative_files2 = np.asarray(sorted(negative_files2))

        dict_negative = {i:[] for i in range(n_files)}
        instances = []

        for file_idx, path in enumerate(csv_files):
            rawfile = open(f"{csv_path}/{path}", "rb").read()
            try:
                file = rawfile.decode("utf-8")
            except UnicodeDecodeError:
                encoding = chardet.detect(rawfile)["encoding"]
                file = rawfile.decode(encoding)

            rows = np.asarray(file.splitlines())
            range_idx = range(min(len(rows), self.max_len))

            count_positive = np.count_nonzero(positive_files == file_idx)
            if count_positive:
                row_idx1 = self.rng.choice(range_idx, size=count_positive, replace=True)
                row_idx2 = self.rng.choice(range_idx, size=count_positive, replace=True)

                for idx, x in enumerate(row_idx1 == row_idx2):
                    if x:
                        new_lst = list(range(0, row_idx2[idx])) + list(range(row_idx2[idx] + 1, len(range_idx)))
                        row_idx2[idx] = self.rng.choice(new_lst, size=1)

                for row1,row2 in zip(rows[row_idx1],rows[row_idx2]):
                    instances.append(self.pair_to_tokens(row1, row2, same_file=True))

            count_negative = np.count_nonzero((negative_files1==file_idx) | (negative_files2==file_idx))
            if count_negative:
                row_idx = list(self.rng.choice(range_idx, size=count_negative, replace=True))
                dict_negative[file_idx] = rows[row_idx]

        for f1_idx, f2_idx in zip(negative_files1, negative_files2):
            row1 = dict_negative[f1_idx].pop()
            row2 = dict_negative[f2_idx].pop()
            instances.append(self.pair_to_tokens(row1, row2, same_file=False))

        instance_df = pd.DataFrame(instances)
        instance_df.rename(columns={0: "same_file"}, inplace=True)
        instance_df.rename(columns={i: f"token_{i - 1}" for i in range(1, self.max_len + 1)}, inplace=True)
        instance_df.rename(columns={i: f"masked_token_{i - self.max_len - 1}" for i in range(self.max_len + 1, self.max_len + self.max_masks + 1)}, inplace=True)
        instance_df.rename(
            columns={i: f"masked_position_{i - self.max_len - self.max_masks - 1}" for i in range(self.max_len + self.max_masks + 1, self.max_len + self.max_masks + self.max_masks + 1)},
            inplace=True)
        # if columns starts with "masked_position", convert to int
        # find if a cell contains null or infinite
        print(instance_df.isnull().values.any())
        print(instance_df.isin([np.inf, -np.inf]).values.any())
        for col in instance_df.columns:
            if col.startswith("masked_position"):
                instance_df[col] = instance_df[col].astype(int)
        instance_path = f"{dataset_path}/instances_max_len{self.max_len}_{'mask_special' if self.mask_special_only else 'mask_all'}.csv"
        instance_df.to_csv(instance_path, index=False, header=True)

        with open(instance_path + "_finished", "w") as f:
            f.write("Processing finished")
        print("Finished materializing instances csv")

    def pair_to_tokens(self,row_a: str, row_b: str = "", same_file: bool = False):
        tokens1 = list(map(str, self.tokenizer(row_a)))
        tokens2 = list(map(str, self.tokenizer(row_b)))

        max_sentence_length = int((self.max_len - 3) / 2)
        tokens1 = tokens1[:max_sentence_length]
        tokens2 = tokens2[:max_sentence_length]

        try:
            assert (len(tokens1) + len(tokens2) <= self.max_len - 3)
        except AssertionError:
            pdb.set_trace()

        tokens = [self.cls_token] + tokens1 + [self.sep_token] + tokens2 + [self.sep_token]
        token_type_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)  # encoding first or second sentence
        padding_mask = [0] * len(tokens)

        if len(tokens) < self.max_len:  # Zero Paddings
            n_pad = self.max_len - len(tokens)
            tokens.extend([self.pad_token] * n_pad)
            token_type_ids.extend([0] * n_pad)
            padding_mask.extend([1] * (n_pad))

        # masking logic
        candidate_masked_positions = []
        for i, token in enumerate(tokens):
            if str(token) != self.cls_token and str(token) != self.sep_token and str(token) != self.pad_token:
                if (not self.mask_special_only) or (self.mask_special_only and contains_special(token)):
                    candidate_masked_positions.append(i)

        masked_tokens, masked_positions = [], []
        n_masks = 0

        if len(candidate_masked_positions):
            n_masks = min(len(candidate_masked_positions), self.max_masks)
            self.rng.shuffle(candidate_masked_positions)
            for pos in candidate_masked_positions[:n_masks]:
                masked_positions.append(pos)
                masked_tokens.append(str(tokens[pos]))
                probability_mask = self.rng.random()
                if probability_mask < 0.8:  # 80%
                    tokens[pos] = self.mask_token  # make mask
                elif 0.8 <= probability_mask <= 0.9:  # 10%
                    tokens[pos] = self.rng.choice(self.vocab)
            for _ in range(n_masks, self.max_masks):
                masked_positions.append(-100)
        else:
            return None

        # Padding masked tokens
        if n_masks < self.max_masks:  # if there are not max_mask masked tokens, pad the masked tokens to reach max_mask
            n_pad = int(self.max_masks - n_masks)
            masked_tokens.extend([self.pad_token] * n_pad)

        return [same_file] + tokens + masked_tokens + masked_positions