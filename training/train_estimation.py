import os
import shutil
from pathlib import Path
import _jsonnet
import json
import sys
import pdb
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath("."))
from csv_embedder.pattern_tokenizer import PatternTokenizer
from csv_embedder.magritte_finetune_estimate.model import MagritteFinetuneEstimation
from csv_embedder.magritte_finetune_estimate.data_module import EstimationDataModule
from csv_embedder.callbacks import PretrainLoaderCallback, TBLogger

MODEL_FOLDER = "results/estimate/"
CONFIG_PATH = "configs/estimate.jsonnet"

config = _jsonnet.evaluate_file(
    CONFIG_PATH,
    ext_vars={"max_len": "128", "encoding_dim": "128"},
)
config = json.loads(config)

tokens = open(config["vocabulary"]["path"]).read().splitlines()
tokens[tokens.index("")] = "\n"
ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
token_vocab = build_vocab(ordered_tokens)
token_vocab.set_default_index(token_vocab["[UNK]"])

model = MagritteFinetuneEstimation(token_vocab=token_vocab, **config["model"])

dm = EstimationDataModule(
    token_vocab=token_vocab, tokenizer=PatternTokenizer(), **config["data_module"]
)

dm.prepare_data()
dm.setup()

logger = TBLogger(**config["logger"])
trainer = pl.Trainer(
    **config["trainer"],
    logger=logger,
    callbacks=[
        PretrainLoaderCallback(**config["callbacks"]["pretrain_loader"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
        ModelCheckpoint(monitoing="val_loss", save_top_k=1,),
    ],
)
trainer.fit(model, dm)
trainer.validate(model, dm)
