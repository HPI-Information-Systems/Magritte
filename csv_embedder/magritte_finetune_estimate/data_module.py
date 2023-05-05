import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from csv_embedder.magritte_finetune_estimate.dataset import EstimationDataset
from csv_embedder.pattern_tokenizer import PatternTokenizer

class EstimationDataModule(pl.LightningDataModule):

    def __init__(self, data_path:str,
                 token_vocab: Vocab,
                 n_files:int,
                 tokenizer:PatternTokenizer,
                 max_rows: int,
                 max_len: int,
                 batch_size:int,
                 num_workers:int,
                 shuffle:bool = True,):
        super().__init__()
        self.data_path = data_path
        self.token_vocab = token_vocab
        self.max_rows = max_rows
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.n_files = n_files
        self.tokenizer = tokenizer
        self.data_train = None
        self.data_val = None
        self.annotations_df = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass


    def _common_dataloader(self, files_path, shuffle=True):
        dataset = EstimationDataset(annotations_df=self.annotations_df,
                                      files_path=files_path,
                                      token_vocab=self.token_vocab,
                                      max_rows=self.max_rows,
                                      max_len=self.max_len,
                                      tokenizer=self.tokenizer, )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        files_path = f"{self.data_path}/dev_test/"
        self.annotations_df = pd.read_csv(f"{files_path}/annotations_train.csv")[:self.n_files]
        return self._common_dataloader(files_path, shuffle=self.shuffle)

    def val_dataloader(self):
        files_path = f"{self.data_path}/dev_test/"
        self.annotations_df = pd.read_csv(f"{files_path}/annotations_dev.csv")[:self.n_files]
        return self._common_dataloader(files_path, shuffle=False)
    
    def test_dataloader(self):
        files_path = f"{self.data_path}/dev_test/"
        self.annotations_df = pd.read_csv(f"{files_path}/annotations_test.csv")[:self.n_files]
        return self._common_dataloader(files_path, shuffle=False)
    