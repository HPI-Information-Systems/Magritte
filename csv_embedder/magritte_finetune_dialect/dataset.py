import os
import traceback

import chardet
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab


class DialectDataset(Dataset):
    def __init__(self, annotations_df,
                 data_path:str,
                 token_vocab: Vocab,
                 label_vocab: Vocab,
                 tokenizer,
                 max_rows=128,
                 max_len=128,
                 n_files=None,
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 for_prediction=False,
                 ):

        self.data_path = data_path
        self.annotations = annotations_df[:n_files].to_dict("records")
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.for_prediction = for_prediction

    def __len__(self):
        return len(self.annotations)

    def read_file(self, filename, delimiter, quotechar, escapechar):
        csv_path = f'{self.data_path}/csv/{filename}'
        if not os.path.isfile(csv_path):
            print(f"File {csv_path} not found, skipping")
            raise FileNotFoundError

        rawfile = open(f"{csv_path}", "rb").read()
        try:
            file = rawfile.decode("utf-8")
        except UnicodeDecodeError:
            encoding = chardet.detect(rawfile)["encoding"]
            file = rawfile.decode(encoding)
        rows = file.splitlines()
        rows = [r for r in rows][:self.max_rows]
        if not self.for_prediction:
            if not os.path.exists(f"{self.data_path}/dialect_tags/{filename}_tags.csv"):
                raise FileNotFoundError("Tag file not found")
            tag_file = open(f"{self.data_path}/dialect_tags/{filename}_tags.csv", "r").read()
            tag_rows = tag_file.splitlines()[:self.max_rows]

            if len(tag_rows) != len(rows):
                if len(rows) > len(tag_rows):
                    diff = len(tag_rows)
                    if rows[:diff] == tag_rows:
                        if rows[diff:] == [''] * (len(rows) - diff):
                            tag_rows += [''] * (len(rows) - diff)
                else:
                    print(f"{filename}: CSV rows: {len(rows)}, Tag rows: {len(tag_rows)}")
                    raise AssertionError
        else:
            tag_rows = None

        if len(rows) < self.max_rows:
            n_padded_rows = (self.max_rows - len(rows))
            rows += [''] * n_padded_rows
            if tag_rows is not None:
                tag_rows += [''] * n_padded_rows

        try:
            target_tags = []
            row_tokens = []
            for i, row in enumerate(rows[:self.max_rows]):
                tokens = []
                tag_tokens = []
                if len(row):
                    tokens = [self.cls_token] + self.tokenizer(row)
                    if tag_rows is not None:
                        tag_tokens = [self.pad_token] + tag_rows[i].split(" ")
                else:
                    tokens = [self.cls_token, self.sep_token]
                    tag_tokens = [self.pad_token, self.pad_token]
                if len(tokens) < self.max_len:  # Zero Paddings
                    n_pad = self.max_len - len(tokens)
                    tokens.extend([self.pad_token] * n_pad)
                    tag_tokens.extend([self.pad_token] * n_pad)
                row_tokens.append(tokens[:self.max_len])
                target_tags.append(tag_tokens[:self.max_len])

            input_tokens = torch.tensor([self.token_vocab(t) for t in row_tokens])
            labels={}
            if not self.for_prediction:
                delm = delimiter[0] if len(delimiter) > 1 else delimiter
                quote = quotechar[0] if len(quotechar) > 1 else quotechar
                escape = escapechar[0] if len(escapechar) > 1 else escapechar

                labels = {"target_tags": torch.tensor([self.label_vocab(t) for t in target_tags])}
                labels["target_delimiter"] = self.token_vocab([delm])[0]
                labels["target_quotechar"] = self.token_vocab([quote])[0]
                labels["target_escapechar"] = self.token_vocab([escape])[0]
            else:
                ann = pd.read_csv(f'{self.data_path}/dialect_annotations.csv').fillna("")
                ann = ann[ann["filename"] == filename]
                labels["target_delimiter"] = ann["delimiter"].values[0]
                labels["target_quotechar"] = ann["quotechar"].values[0]
                labels["target_escapechar"] = ann["escapechar"].values[0]

            assert len(input_tokens) == self.max_len
            return input_tokens, labels

        except Exception as e:
                traceback.print_exc()
                print(f"Reader exception: {e}")
                print(f"Filename: {filename}")

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        filename = ann["filename"]
        delimiter = ann["delimiter"]
        quotechar = ann["quotechar"]
        escapechar = ann["escapechar"]

        input_tokens, labels = self.read_file(filename, delimiter, quotechar, escapechar)
        return input_tokens, labels


