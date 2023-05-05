import _jsonnet
import json
import sys
sys.path.append(".")
sys.path.append("..")
from csv_embedder.pattern_tokenizer import PatternTokenizer
from csv_embedder.magritte_pretrain_ae.model import MagrittePretrainingVAE
from csv_embedder.magritte_pretrain_ae.data_module import CsvFileDataModule
from lightning.pytorch.callbacks import RichProgressBar

from lightning.pytorch.loggers import TensorBoardLogger
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from csv_embedder.callbacks import PretrainLoaderCallback

MODEL_FOLDER = "results/pretrain_ae/"

CONFIG_PATH = "configs/ae.jsonnet"

config = _jsonnet.evaluate_file(CONFIG_PATH)
config = json.loads(config)

tokens = open(config["vocabulary"]["path"]).read().splitlines()
tokens[tokens.index("")] = "\n"
ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
token_vocab = build_vocab(ordered_tokens)
token_vocab.set_default_index(token_vocab["[UNK]"])

model = MagrittePretrainingVAE(token_vocab=token_vocab, **config["model"])

dm = CsvFileDataModule(
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
        RichProgressBar(),
    ],
    limit_val_batches=0,
)
trainer.fit(model, dm)
