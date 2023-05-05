import os
import _jsonnet
import json
import sys
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint

sys.path.append(os.path.abspath("."))
from csv_embedder.pattern_tokenizer import PatternTokenizer
from csv_embedder.magritte_finetune_dialect.model import MagritteFinetuneDialectDetection
from csv_embedder.magritte_finetune_dialect.data_module import DialectDataModule
from csv_embedder.callbacks import PretrainLoaderCallback, TBLogger

MODEL_FOLDER = "results/dialect/"

CONFIG_PATH = "configs/dialect.jsonnet"

config = _jsonnet.evaluate_file(CONFIG_PATH,
                                ext_vars={"max_len": "128",
                                            "encoding_dim": "128"})
config = json.loads(config)

tokens = open(config["vocabulary"]["directory"] + "/tokens.txt").read().splitlines()
tokens[tokens.index("")] = "\n"
ordered_tokens =  {t:len(tokens)-i for i,t in enumerate(tokens)}
token_vocab = build_vocab(ordered_tokens)
token_vocab.set_default_index(token_vocab["[UNK]"])
dialect_classes = open(config["vocabulary"]["directory"] + "/dialect_labels.txt").read().splitlines()
ordered_classes = {c:len(dialect_classes)-i for i,c in enumerate(dialect_classes)}
label_vocab = build_vocab(ordered_classes)

model = MagritteFinetuneDialectDetection(token_vocab=token_vocab,
                                            label_vocab=label_vocab,
                                            **config["model"])

dm = DialectDataModule(
    token_vocab=token_vocab,
    label_vocab=label_vocab,
    tokenizer=PatternTokenizer(),
    **config["data_module"]
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
        RichProgressBar(),
        ModelCheckpoint(monitor='val_loss', save_top_k=1,)
    ],
)
trainer.fit(model, dm)
