from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os

MAX_LEN = 512
SEP_TOKEN_ID = 102


class TableDataset(Dataset):
    # code from train/test
    def __init__(
        self, jsonl_path, tokenizer, label_dict, save_path=None, *args, **kwargs
    ):

        self.tokenizer = tokenizer
        self.jsonl_path = jsonl_path
        self.label_dict = label_dict

        if os.path.exists(save_path):
            (
                self.filenames,
                self.datasets,
                self.data,
                self.labels,
                self.rel,
                self.sub,
                self.labels,
            ) = torch.load(save_path)

        else:
            (
                self.filenames,
                self.datasets,
                self.data,
                self.labels,
                self.rel,
                self.sub,
                self.labels,
            ) = self.tokenize_and_save(save_path)

    def tokenize_and_save(self, save_path):

        self.filenames = []
        self.datasets = []
        self.data = []
        self.labels = []
        self.rel = []
        self.sub = []
        self.labels = []

        with open(self.jsonl_path, "r+", encoding="utf8") as jl:
            dicts = [json.loads(line) for line in jl.readlines()]
        for item in tqdm(dicts, desc="Tokenizing"):
            self.filenames.append(item["filename"])
            self.datasets.append(item["dataset"])

            label_idx = int(self.label_dict[item["label"]])
            self.labels.append(torch.tensor(label_idx))

            target_data = np.array(item["content"])[:, int(item["target"])]
            data = ""
            for i, cell in enumerate(target_data):
                data += cell
                data += " "
            target_token_ids = self.tokenize(data)
            self.data.append(target_token_ids)

            cur_rel_cols = [np.array(col) for col in item["related_cols"]]
            if not cur_rel_cols:
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(cur_rel_cols)
            self.rel.append(rel_token_ids)

            cur_sub_rel_cols = [np.array(col) for col in item["sub_related_cols"]]
            if not cur_sub_rel_cols:
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(cur_sub_rel_cols)
            self.sub.append(sub_token_ids)

        state = [
            self.filenames,
            self.datasets,
            self.data,
            self.labels,
            self.rel,
            self.sub,
            self.labels,
        ]
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(state, save_path)
        return state

    def tokenize(self, text):
        tokenized_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
        )
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set(self, cols):
        text = ""
        for i, col in enumerate(cols):
            for cell in col:
                text += cell
                text += " "
            if not i == len(cols) - 1:
                text += "[SEP]"
        tokenized_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
        )
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set_equal(
        self, cols
    ):  # Assigning the tokens equally to each identified column
        init_text = ""
        for i, col in enumerate(cols):
            for cell in col:
                init_text += cell
                init_text += " "
            if not i == len(cols) - 1:
                init_text += "[SEP]"
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= MAX_LEN:
            tokenized_text = self.tokenizer.encode_plus(
                init_text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
            )
        else:
            ratio = MAX_LEN / total_length
            text = ""
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col) * ratio:
                        break
                    text += cell
                    text += " "
                if not i == len(cols) - 1:
                    text += "[SEP]"
            tokenized_text = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
            )
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        return self.data[idx], self.rel[idx], self.sub[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels
