import os
import shutil
from pathlib import Path
import _jsonnet
import json
import sys
import pdb

from lightning.pytorch.loggers import TensorBoardLogger
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

sys.path.append(os.path.abspath("."))
from csv_embedder.pattern_tokenizer import PatternTokenizer
from csv_embedder.magritte_pretrain_rowpair.model import MagrittePretrainingRowPair
from csv_embedder.magritte_pretrain_rowpair.data_module import RowPairDataModule
from csv_embedder.callbacks import PretrainLoaderCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_step", type=int, default=1)
args = parser.parse_args()
pretrain_step = args.pretrain_step

MODEL_FOLDER = f"results/pretrain_rowpair_{pretrain_step}/"
if os.path.exists(MODEL_FOLDER + "tensorboard/lightning_logs"):
    shutil.rmtree(MODEL_FOLDER + "tensorboard/lightning_logs")

if os.path.exists(MODEL_FOLDER + "model"):
    shutil.rmtree(MODEL_FOLDER + "model")


CONFIG_PATH = f"configs/rowpair_{pretrain_step}.jsonnet"

config = _jsonnet.evaluate_file(CONFIG_PATH)
config = json.loads(config)

tokens = open(config["vocabulary"]["path"]).read().splitlines()
tokens[tokens.index("")] = "\n"
ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
token_vocab = build_vocab(ordered_tokens)
token_vocab.set_default_index(token_vocab["[UNK]"])

model = MagrittePretrainingRowPair(vocab=token_vocab, **config["model"])

dm = RowPairDataModule(
    token_vocab=token_vocab,
    tokenizer=PatternTokenizer(),
    **config["data_module"])

dm.prepare_data()
dm.setup()

logger = TensorBoardLogger(**config["logger"])
trainer = pl.Trainer(
    **config["trainer"],
    logger=logger,
    callbacks=[
        PretrainLoaderCallback(**config["callbacks"]["pretrain_loader"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
    ],
)

trainer.fit(model, dm)