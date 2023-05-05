import csv
import json
import os
import io
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

class RowClassDataset(Dataset):
    def __init__(self, data_path,
                 token_vocab: Vocab,
                 label_vocab: Vocab,
                 tokenizer,
                 max_rows=10,
                 max_len=32,
                 n_files=None,
                 ):

        self.data_path = data_path
        jsonlines = open(self.data_path).read().splitlines()
        self.annotations = list(x for x in map(json.loads, jsonlines))
        self.annotations = self.annotations[:n_files]
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations)

    def get_groups_indices(self):
        groups = {}
        for idx, ann in enumerate(self.annotations):
            group = ann["group"]
            if group not in groups:
                groups[group] = []
            groups[group].append(idx)
        return groups

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        filepath = f"{os.path.dirname(self.data_path)}/{ann['group']}/{ann['filename']}"

        annotations = ann["line_annotations"]
        empty_idx = [idx for idx, x in enumerate(annotations) if x == "empty"]
        reader = csv.reader(open(filepath, "r", newline=''), delimiter=",", quotechar='"')

        all_rows = []
        all_classes = []
        for idx, row in enumerate(reader):
            if idx in empty_idx:
                continue
            else:
                rowbuf = io.StringIO()
                csv.writer(rowbuf, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL).writerow(row)
                all_rows.append(rowbuf.getvalue()[:-2].replace("\r\n", "\n"))
                all_classes.append(annotations[idx])
        rows = []
        row_classes = []
        if len(all_rows) > self.max_rows:
            data_idx = [idx for idx, x in enumerate(all_classes) if x == "data"]
            nondata_idx = [idx for idx, x in enumerate(all_classes) if x != "data"]
            if len(nondata_idx) < self.max_rows-1:
                sampled_idx = data_idx[:(self.max_rows-len(nondata_idx))]
                for idx,r in enumerate(all_rows):
                    if idx in nondata_idx+list(sampled_idx):
                        rows.append(r)
                        row_classes.append(all_classes[idx])
            else:
                rows = all_rows[:self.max_rows]
                row_classes = all_classes[:self.max_rows]
        else:
            rows = all_rows
            row_classes = all_classes
        if len(rows) < self.max_rows:
            n_padded_rows = (self.max_rows - len(rows))
            rows += [''] * n_padded_rows
            row_classes += ['empty'] * n_padded_rows

        try:
            assert len(row_classes) == len(rows), f"Number of rows and number of row classes are not the same: {len(row_classes)} != {len(rows)}"
            row_tokens = []
            for row in rows:
                if not len(row):
                    tokens=["[CLS]","[SEP]"]
                else:
                    tokens = ["[CLS]"] + self.tokenizer(row)
                if len(tokens) < self.max_len:  # Zero Paddings
                    n_pad = self.max_len - len(tokens)
                    tokens.extend(["[PAD]"] * n_pad)
                row_tokens.append(tokens[:self.max_len])

            input_tokens = torch.tensor([self.token_vocab(t) for t in row_tokens])
            row_labels = torch.tensor(self.label_vocab(row_classes))

            assert len(row_tokens) == len(row_classes)
            assert set(row_classes) != set(["empty"]), f"Row classes are all empty: {row_classes}"
            return input_tokens, row_labels

        except Exception as e:
                print(f"Reader exception: {e}")
                print(f"Filename: {filepath}")
