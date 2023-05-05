import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab


class RowPairDataset(Dataset):
    def __init__(self, instance_df,
                 data_path: str,
                 token_vocab: Vocab,
                 tokenizer,
                 max_rows=10,
                 max_len=32,
                 n_pairs=None,
                 max_percent=0.15,
                 ):
        self.data_path = data_path
        self.instances = instance_df[:n_pairs]
        self.token_vocab = token_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.max_percent = max_percent
        self.max_masks = int(np.ceil(self.max_percent * max_len))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        pair = dict(self.instances.iloc[idx])
        tokens = [pair[f"token_{idx}"] for idx in range(self.max_len)]
        masked_tokens = [pair[f"masked_token_{idx}"] for idx in range(self.max_masks)]
        masked_positions = [pair[f"masked_position_{idx}"] for idx in range(self.max_masks)]
        same_file = pair["same_file"]

        sep_index1 = tokens.index("[SEP]")
        sep_index2 = tokens.index("[SEP]", sep_index1 + 1)
        token_type_ids = [0] * (sep_index1 + 1)  # +1 because we want to include the CLS token
        token_type_ids += [1] * (sep_index2 - sep_index1)
        token_type_ids += [0] * (self.max_len - len(token_type_ids))  # padding gets 0

        input_tokens = torch.tensor(self.token_vocab(tokens))
        masked_tokens = torch.tensor(self.token_vocab(masked_tokens))
        masked_positions = torch.tensor(masked_positions)
        token_type_ids = torch.tensor(token_type_ids)
        same_file = torch.tensor(same_file)

        return {"input_tokens": input_tokens,
                "token_type_ids": token_type_ids,
                "masked_positions": masked_positions}, \
            {"masked_tokens": masked_tokens, "same_file": same_file}
