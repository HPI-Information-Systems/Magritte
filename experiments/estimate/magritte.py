import os
import sys

sys.path.append(".")
sys.path.append("..")

import json
import _jsonnet  # type: ignore
import lightning.pytorch as pl  # type: ignore
import pandas as pd
from torch.utils.data import DataLoader  # type: ignore
from torchtext.vocab import vocab as build_vocab  # type: ignore
from sklearn import metrics

from csv_embedder import PatternTokenizer
from csv_embedder.magritte_finetune_estimate.data_module import EstimationDataModule
from csv_embedder.magritte_finetune_estimate.model import MagritteFinetuneEstimation

import logging

logging.getLogger("lightning").setLevel(0)


class MagritteEvaluator():
    def __init__(
        self,
        data_dir: str,
        sys_name: str,
        experiment_dir: str,
        results_file: str,
        config_path="configs/estimate.jsonnet",
        weights_path="",
        global_max=True,
        cuda_device=-1,
        batch_size=16,
        n_repetitions: int = 1,
        n_workers=64,
        subset: int = None,
        skip_processing=False,
        *args,
        **kwargs,
    ):
        self.n_repetitions = n_repetitions
        self.subset = subset
        self.data_dir = data_dir
        self.skip_processing = skip_processing
        self.n_workers = n_workers
        self.results_file = results_file
        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        config = _jsonnet.evaluate_file(self.config_path)
        config = json.loads(config)
        self.batch_size = batch_size
        config["trainer"]["devices"] = [self.cuda_device]

        self.max_rows = config["data_module"]["max_rows"]
        self.max_len = config["data_module"]["max_len"]

        tokens = (
            open(config["vocabulary"]["path"]).read().splitlines()
        )
        ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
        self.token_vocab = build_vocab(ordered_tokens)
        self.token_vocab.set_default_index(self.token_vocab["[UNK]"])

        self.magritte = MagritteFinetuneEstimation(
            token_vocab=self.token_vocab,
            **config["model"],
        )
        self.magritte.load_weights(weights_path)
        self.magritte.eval()

        config["data_module"]["shuffle"] = False
        config["data_module"]["batch_size"] = batch_size
        self.trainer = pl.Trainer(**config["trainer"])
        self.dm = EstimationDataModule(
            token_vocab=self.token_vocab,
            tokenizer=PatternTokenizer(),
            **config["data_module"],
        )
        self.dm.setup("dev")

    def print_results(self):
        print("null")

    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print(
                f"Results file already exists in {self.results_file}, skipping processing"
            )
            return

        dl = self.dm.test_dataloader()
        annotations = self.dm.annotations_df.to_dict("records")

        batches = self.trainer.predict(self.magritte, dl)
        results=[]
        for batch_idx, predicted in enumerate(batches):
            for file_idx, estimate in enumerate(predicted):
                filename_idx = (batch_idx * self.batch_size) + file_idx
                results.extend(
                    [
                        {
                            "filename": annotations[filename_idx]["source"],
                            "estimate": estimate.item(),
                            "target": annotations[filename_idx]["pollution_level"],
                            # "unique_length": annotations[filename_idx]["unique_length"],
                            # "script_length": annotations[filename_idx]["script_length"],
                        }
                    ]
                )

        results_df = pd.DataFrame.from_dict(results)

        results_df.to_csv(self.results_file, index=False)
        print("Saving the results to ", self.results_file)
        print("MSE: ", metrics.mean_squared_error(results_df["target"], results_df["estimate"]))


if __name__ == "__main__":
    n_reps = 3

    for rep in range(n_reps):
        evaluator = MagritteEvaluator(
            data_dir=f"data/estimate/",
            sys_name=f"magritte_{rep}",
            experiment_dir="experiments/estimate/",
            results_file=f"experiments/estimate/magritte_{rep}_extended_results.csv",
            skip_processing=False,
            batch_size=1,
            subset=100,
            n_workers=100,
            cuda_device=1,
            weights_path = "weights/magritte_estimate.pth",
        )
        evaluator.evaluate()
        res = evaluator.print_results()