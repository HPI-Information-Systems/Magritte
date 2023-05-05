import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import Vocab

from csv_embedder.magritte_finetune_rowclass.dataset import RowClassDataset
from csv_embedder.pattern_tokenizer import PatternTokenizer


class RowClassDataModule(pl.LightningDataModule):

    def __init__(self, data_path:str,
                 token_vocab: Vocab,
                 label_vocab: Vocab,
                 n_files:int,
                 tokenizer:PatternTokenizer,
                 max_rows: int,
                 max_len: int,
                 batch_size:int,
                 num_workers:int,
                 train_datasets:list,
                 val_dataset_name:str,
                 shuffle:bool = True,):
        super().__init__()
        self.data_path = data_path
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_datasets = train_datasets
        self.val_dataset_name = val_dataset_name
        self.shuffle = shuffle
        self.n_files = n_files
        self.tokenizer = tokenizer
        self.data_train = None
        self.data_val = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            self.dataset_full = RowClassDataset(self.data_path,
                                            token_vocab = self.token_vocab,
                                            label_vocab = self.label_vocab,
                                            max_rows = self.max_rows,
                                            max_len = self.max_len,
                                            tokenizer = self.tokenizer,
                                            n_files = self.n_files,)
            get_groups_indices = self.dataset_full.get_groups_indices()
            self.val_indices = get_groups_indices[self.val_dataset_name]
            self.train_indices = []
            for d in self.train_datasets:
                self.train_indices.extend(get_groups_indices[d])
            self.data_train, self.data_val = Subset(self.dataset_full, self.train_indices), Subset(self.dataset_full, self.val_indices)

    def _common_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._common_dataloader(self.data_train, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._common_dataloader(self.data_val, shuffle=False)

    def full_dataloader(self):
        return self._common_dataloader(self.dataset_full, shuffle=False)
