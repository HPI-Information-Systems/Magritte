import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("related/rowclass/")

from csv_embedder.magritte_finetune_rowclass.model import (
    MagritteFinetuneRowClassification,
)
from csv_embedder.magritte_finetune_rowclass.data_module import LineClassDataModule
from evaluator import Evaluator
from strudel_lib.lstrudel import LStrudel
from torchtext.vocab import vocab as build_vocab  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import lightning.pytorch as pl  # type : ignore
import torch.nn as nn  # type : ignore
import torch  # type : ignore
import _jsonnet  # type : ignore
import pandas as pd
import csv
import numpy as np
import os.path
from pqdm.processes import pqdm  # type : ignore
from csv_embedder.pattern_tokenizer import PatternTokenizer
import json
import os


def load_data(file_path):
    reader = csv.reader(open(file_path, "r"))
    return list(reader)


class StrugritteEvaluator(Evaluator):
    def __init__(
        self,
        model_file,
        weights_path="",
        global_max=True,
        cuda_device=-1,
        config_path="configs/lineclass.jsonnet",
        batch_size=16,
        use_row_cls=True,
        use_lineprobs=True,
        strudel_features_file="",
        magritte_features_file="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_file = model_file

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device
        self.use_row_cls = use_row_cls
        self.use_lineprobs = use_lineprobs
        self.strudel_features_file = strudel_features_file
        self.magritte_features_file = magritte_features_file

        print(self.train_datasets)
        config = _jsonnet.evaluate_file(
            self.config_path,
            ext_vars={
                "validation_dataset": self.test_dataset,
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

        self.ls = LStrudel(n_jobs=self.n_workers)

        config["data_module"]["shuffle"] = False
        config["data_module"]["batch_size"] = self.batch_size
        config["data_module"]["num_workers"] = self.n_workers
        self.trainer = pl.Trainer(**config["trainer"])
        self.dm = LineClassDataModule(
            token_vocab=self.token_vocab,
            label_vocab=self.label_vocab,
            tokenizer=PatternTokenizer(),
            **config["data_module"],
        )
        self.dm.setup()

    def strudel_features(self, files, annotation_file, *args, **kwargs):
        data = pqdm(files, load_data, n_jobs=self.n_workers, desc="Loading files")
        annotation_dicts = [
            json.loads(r) for r in open(annotation_file, "r").read().splitlines()
        ]
        annotations = {d["filename"]: d["line_annotations"] for d in annotation_dicts}

        args = [
            {
                "filename": os.path.basename(f),
                "table_array": data[idx],
                "line_labels": annotations[os.path.basename(f)],
                "max_rows": 128,
                "sample_data": True,
            }
            for idx, f in enumerate(files)
        ]
        features = pqdm(
            args,
            self.ls.create_line_feature_vector,
            argument_type="kwargs",
            n_jobs=self.n_workers,
            desc="Creating traning features",
        )
        return pd.concat(features)

    def magritte_features(self, dl, *args, **kwargs):
        batches = self.trainer.predict(self.magritte, dl)
        filenames = [x["filename"] for x in self.dm.dataset_full.annotations]
        results = []
        filename_idx = 0
        for batch in batches:
            y, target = batch
            predicted = (
                nn.Softmax(dim=2)(y["lineclass_logits"].type(torch.double))
                .cpu()
                .numpy()
            )
            target = target.cpu().numpy()
            for file_batch_idx, line_prediction in enumerate(predicted):
                if self.magritte.empty_index in target[file_batch_idx]:
                    num_lines =  target[file_batch_idx].tolist().index(self.magritte.empty_index)
                else:
                    num_lines = self.magritte.max_rows
                for line_idx, logits in enumerate(line_prediction[:num_lines]):
                    results += [
                        {
                            "filename": filenames[filename_idx],
                            "line_number": line_idx,
                            "row_embeddings": y["row_embeddings"][
                                file_batch_idx, :, line_idx, 0
                            ]
                            .cpu()
                            .numpy(),
                            "data_logits": logits[0],
                            "derived_logits": logits[1],
                            "group_logits": logits[2],
                            "header_logits": logits[3],
                            "metadata_logits": logits[4],
                            "notes_logits": logits[5],
                        }
                    ]
                filename_idx += 1
        return pd.DataFrame(results)

    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print(f"File {self.results_file} exists, skipping processing")
            return

        if not os.path.exists(self.strudel_features_file):
            files = []
            for t in set(self.train_datasets + [self.test_dataset, "mendeley", "troy"]):
                for f in os.listdir(os.path.join(self.data_dir, t)):
                    files.append(f"{self.data_dir}/{t}/{f}")

            annotation_file = self.data_dir + "strudel_annotations.jsonl"
            strudel_df = self.strudel_features(files, annotation_file)
            strudel_df.to_csv(self.strudel_features_file, index=False)
        else:
            strudel_df = pd.read_csv(self.strudel_features_file)
        
        if not os.path.exists(self.magritte_features_file):
            dl = self.dm.full_dataloader()
            magritte_df = self.magritte_features(dl)
            magritte_df = magritte_df.join(
                pd.DataFrame(
                    magritte_df["row_embeddings"].tolist(),
                    index=magritte_df.index,
                    columns=[f"row_embedding_{i}" for i in range(768)],
                )
            )
            del magritte_df["row_embeddings"]
            magritte_df.to_csv(self.magritte_features_file, index=False)
        else:
            magritte_df = pd.read_csv(self.magritte_features_file)

        if not self.use_row_cls:
            magritte_df = magritte_df[
                [c for c in magritte_df.columns if not c.startswith("row_embedding")]
            ]

        if not self.use_lineprobs:
            magritte_df = magritte_df[
                [c for c in magritte_df.columns if not c.endswith("_logits")]
            ]

        features_df = pd.merge(strudel_df, magritte_df, on=["filename", "line_number"])
        features_df = features_df[features_df["label"] != "empty"].reset_index(
            drop=True
        )

        train_files = [
            f for d in self.train_datasets for f in os.listdir(self.data_dir + d)
        ]
        train_df = features_df[features_df["filename"].isin(train_files)].reset_index(
            drop=True
        )

        print("Training model...")
        self.ls.fit(train_df)
        test_files = [f for f in os.listdir(self.data_dir + self.test_dataset)]
        test_df = features_df[features_df["filename"].isin(test_files)].reset_index(
            drop=True
        )

        tmp_results = []
        print("Predicting model...")
        result = self.ls.predict(test_df)
        tmp_results.append(result)

        results_df = pd.concat(tmp_results, axis=0)
        results_df.to_csv(self.results_file)
        self.ls.save(self.model_file)


if __name__ == "__main__":
    results = {}
    train_datasets = ["deex", "saus", "cius", "govuk"]
    for r, l in [(False, True), (False, False)]:
        for dataset in ["mendeley", "troy"] + train_datasets:
            print("Using row cls", r, "and line probs", l)
            print(
                "test dataset",
                dataset,
            )
            evaluator = StrugritteEvaluator(
                data_dir=f"data/line_classification/",
                sys_name=f"strugritterow_{r}_line_{l}",
                experiment_dir="experiments/finetune_lineclass/",
                cuda_device=0,
                weights_path=f"weights/finetune_lineclass/magritte_lineclass{dataset}.pth",
                train_datasets=[x for x in train_datasets if x != dataset],
                test_dataset=dataset,
                skip_processing=False,
                subset=None,
                n_workers=100,
                model_file=f"weights/finetune_lineclass/strugritteval_{dataset}.pkl",
                use_row_cls=r,
                use_lineprobs=l,
                batch_size=50,
                strudel_features_file=f"weights/finetune_lineclass/strudel_features.csv",
                magritte_features_file=f"weights/finetune_lineclass/magrittefeatures_{dataset}.csv",
            )
            evaluator.evaluate()
            res = evaluator.print_results()
            (
                data_f1,
                header_f1,
                metadata_f1,
                group_f1,
                derived_f1,
                notes_f1,
                accuracy,
                macro_avg,
            ) = res
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
        pd.DataFrame.from_dict(results, orient="index", columns=["strugritte"]).to_csv(
            f"plots/results/strugritteresults_row_{r}_line_{l}.csv",
            index_label="measure",
        )
