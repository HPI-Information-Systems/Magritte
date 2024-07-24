import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab


class CsvFileDataset(Dataset):
    def __init__(self, csv_files,
                 data_path: str,
                 token_vocab: Vocab,
                 tokenizer,
                 max_rows=10,
                 max_len=32,
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 ):
        self.csv_files = csv_files
        self.data_path = data_path
        self.token_vocab = token_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        filename = self.csv_files[idx]
        csv_path = f"{self.data_path}/{filename}"

        rawdata = open(csv_path, "rb").read()
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


        input_tokens = torch.tensor([self.token_vocab(t) for t in row_tokens])
        return input_tokens