import os
import _jsonnet
import json
import sys

from torchtext.vocab import vocab
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

sys.path.append(os.path.abspath("."))
from csv_embedder.pattern_tokenizer import PatternTokenizer
from csv_embedder.magritte_finetune_rowclass.model import MagritteFinetuneRowClassification
from csv_embedder.magritte_finetune_rowclass.data_module import RowClassDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from csv_embedder.callbacks import PretrainLoaderCallback, OverrideEpochStepCallback

from lightning.pytorch.loggers import TensorBoardLogger

MODEL_FOLDER = "results/rowclass/"

train_datasets = ["saus", "cius","govuk","deex"]
for validation_dataset in ["saus", "cius","govuk", "deex"]:
    CONFIG_PATH = f"configs/lineclass.jsonnet"
    config = _jsonnet.evaluate_file(CONFIG_PATH,
                                    ext_vars={"validation_dataset": validation_dataset})
    config = json.loads(config)

    tokens = open(config["vocabulary"]["directory"] + "/tokens.txt").read().splitlines()
    tokens[tokens.index("")] = "\n"
    token_vocab = vocab({token: 1 for token in tokens})
    line_classes = open(config["vocabulary"]["directory"] + "/lineclass_labels.txt").read().splitlines()
    label_vocab = vocab({lineclass: 1 for lineclass in line_classes})

    model = MagritteFinetuneRowClassification(token_vocab=token_vocab,
                                            label_vocab=label_vocab,
                                            **config["model"])

    dm = RowClassDataModule(
        token_vocab=token_vocab,
        label_vocab=label_vocab,
        tokenizer=PatternTokenizer(),
        **config["data_module"]
    )

    dm.prepare_data()
    dm.setup()

    callbacks=[
        PretrainLoaderCallback(**config["callbacks"]["pretrain_loader"]),            
        OverrideEpochStepCallback(),
        ]
    if validation_dataset in ("mendeley", "troy"):
        limit_val_batches = 0
    else:
        limit_val_batches = None
        callbacks += [EarlyStopping(**config["callbacks"]["early_stopping"]),
                    ModelCheckpoint(monitor="val_loss", save_top_k=1,),]
        
    logger = TensorBoardLogger(**config["logger"], name="val_"+validation_dataset)
    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        limit_val_batches=limit_val_batches,
        callbacks = callbacks,
    )
    trainer.fit(model, dm)
    if validation_dataset != "full":
        trainer.validate(model, dm)
    
    model.save_path = config["model"]["save_path"]
    model.save_weights()
