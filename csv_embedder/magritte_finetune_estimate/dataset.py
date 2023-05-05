import os
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

class EstimationDataset(Dataset):
    def __init__(self, annotations_df,
                 files_path:str,
                 token_vocab: Vocab,
                 tokenizer,
                 max_rows=128,
                 max_len=128,
                 n_files=None,
                 for_prediction=False,
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 ):

        self.files_path = files_path
        self.annotations = annotations_df[:n_files].to_dict("records")
        self.token_vocab = token_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.for_prediction = for_prediction

        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token

    def __len__(self):
        return len(self.annotations)

    def read_file(self, filepath):
        rawdata = open(filepath, "rb").read()
        try:
            rows = rawdata.decode("utf-8").splitlines()[:self.max_rows]
        except UnicodeDecodeError:
            rows = rawdata.decode("latin-1").splitlines()[:self.max_rows]
        if len(rows) < self.max_rows:
            rows += [''] * (self.max_rows - len(rows))

        row_tokens = []
        for r in rows:
            if len(r):
                tokens = [self.cls_token] + self.tokenizer(r)
            else:
                tokens = [self.cls_token] + [self.sep_token]
            if len(tokens) < self.max_len:  # Zero Paddings
                n_pad = self.max_len - len(tokens)
                tokens.extend([self.pad_token] * n_pad)
            row_tokens.append(tokens[:self.max_len])

        return torch.tensor([self.token_vocab(t) for t in row_tokens])


    def __getitem__(self, idx):
        ann = self.annotations[idx]
        target = ann["pollution_level"]
        filepath_a = f"{os.path.dirname(self.files_path)}/csv/{ann['source']}"
        filepath_b = f"{os.path.dirname(self.files_path)}/csv/{ann['target']}"

        input_tokens_a = self.read_file(filepath_a)
        input_tokens_b = self.read_file(filepath_b)
        return input_tokens_a, input_tokens_b, target