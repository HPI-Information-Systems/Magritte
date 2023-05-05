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
from csv_embedder.magritte_finetune_rowclass.data_module import LineClassDataModule
from csv_embedder.magritte_finetune_rowclass.model import (
    MagritteFinetuneRowClassification,
)


from evaluator import Evaluator

import logging

logging.getLogger("lightning").setLevel(0)


class MagritteEvaluator(Evaluator):
    def __init__(
        self,
        config_path="configs/lineclass.jsonnet",
        weights_path="",
        global_max=True,
        cuda_device=-1,
        batch_size=16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        config = _jsonnet.evaluate_file(
            self.config_path,
            ext_vars={
                "validation_dataset": self.test_dataset,
                "max_len": "128",
                "encoding_dim": "128",
            },
        )
        config = json.loads(config)
        self.batch_size = batch_size
        config["trainer"]["devices"] = [self.cuda_device]

        self.max_rows = config["data_module"]["max_rows"]
        self.max_len = config["data_module"]["max_len"]

        tokens = (
            open(config["vocabulary"]["directory"] + "/tokens.txt").read().splitlines()
        )
        ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
        self.token_vocab = build_vocab(ordered_tokens)
        self.token_vocab.set_default_index(self.token_vocab["[UNK]"])

        line_classes = (
            open(config["vocabulary"]["directory"] + "/lineclass_labels.txt")
            .read()
            .splitlines()
        )
        self.label_vocab = build_vocab({lineclass: 1 for lineclass in line_classes})

        self.magritte = MagritteFinetuneRowClassification(
            token_vocab=self.token_vocab,
            label_vocab=self.label_vocab,
            **config["model"],
        )
        self.magritte.load_weights(weights_path)
        self.magritte.eval()

        config["data_module"]["shuffle"] = False
        config["data_module"]["batch_size"] = batch_size
        self.trainer = pl.Trainer(**config["trainer"])
        self.dm = LineClassDataModule(
            token_vocab=self.token_vocab,
            label_vocab=self.label_vocab,
            tokenizer=PatternTokenizer(),
            **config["data_module"],
        )
        self.dm.setup()

    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print(
                f"Results file already exists in {self.results_file}, skipping processing"
            )
            return

        dl = self.dm.val_dataloader()
        batches = self.trainer.predict(self.magritte, dl)
        lookup_vocab = {i: k for i, k in enumerate(self.label_vocab.get_itos())}
        filenames = [
            self.dm.dataset_full.annotations[idx]["filename"]
            for idx in self.dm.val_indices
        ]
        anns = [
            self.dm.dataset_full.annotations[idx]["line_annotations"]
            for idx in self.dm.val_indices
        ]

        results = []
        batch_size = len(batches[0])
        for batch_idx, batch in enumerate(batches):
            y, target = batch
            predicted = y["lineclass"].cpu().numpy()
            target = target.cpu().numpy()
            for file_idx, line_prediction in enumerate(predicted):
                filename_idx = (batch_idx * batch_size) + file_idx
                num_lines = len(anns[filename_idx]) # sometimes there is padding
                results.extend(
                    [
                        { 
                            "filename": filenames[filename_idx],
                            "line_number": line_idx,
                            "predicted": lookup_vocab[line],
                            "label": lookup_vocab[target[file_idx, line_idx]],
                        }
                        for line_idx, line in enumerate(line_prediction[:num_lines])
                    ]
                )

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(self.results_file)
        print("Saving the results to ", self.results_file)

if __name__ == "__main__":
    n_reps = 3
    results = {}
    train_datasets = ["saus", "cius", "deex", "govuk"]
    for dataset in ["mendeley", "troy"]:
        data_f1 = []
        header_f1 = []
        metadata_f1 = []
        group_f1 = []
        derived_f1 = []
        notes_f1 = []
        accuracy = []
        macro_avg = []
        
        for rep in range(n_reps):
            print("Validating on dataset", dataset)
            evaluator = MagritteEvaluator(
                data_dir=f"data/line_classification/",
                sys_name=f"magritte{rep}",
                experiment_dir="experiments/finetune_lineclass/",
                train_datasets=[x for x in train_datasets if x != dataset],
                test_dataset=dataset,
                skip_processing=False,
                batch_size=80,
                subset=None,
                n_workers=100,
                cuda_device=0,
                weights_path=f"weights/finetune_lineclass/magritte_lineclass{dataset}.pth",
            )
            evaluator.evaluate()
            res = evaluator.print_results()
            da_f1, h_f1, m_f1, g_f1, e_f1, no_f1, acc, m_avg = res
            data_f1.append(da_f1)
            header_f1.append(h_f1)
            metadata_f1.append(m_f1)
            group_f1.append(g_f1)
            derived_f1.append(e_f1)
            notes_f1.append(no_f1)
            accuracy.append(acc)
            macro_avg.append(m_avg)

        results.update(
            {
                f"{dataset}_data_f1": data_f1,
                f"{dataset}_header_f1": header_f1,
                f"{dataset}_metadata_f1": metadata_f1,
                f"{dataset}_group_f1": group_f1,
                f"{dataset}_derived_f1": derived_f1,
                f"{dataset}_notes_f1": notes_f1,
                f"{dataset}_accuracy": accuracy,
                f"{dataset}_macro_avg": macro_avg,
            }
        )

        pd.DataFrame.from_dict(results, orient="index", columns=[f"magritte_{rep}" for rep in range(n_reps)]).to_csv(
            "plots/results/magritteresults.csv", index_label="measure"
        )
