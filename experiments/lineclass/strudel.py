import json
import os
import pdb
import tqdm
from pqdm.processes import pqdm
import os.path
import numpy as np
import csv
import pandas as pd
import sys

sys.path.append("related/lineclass/")
from strudel_lib.lstrudel import LStrudel
from evaluator import Evaluator


def load_data(file_path):
    reader = csv.reader(open(file_path, "r"))
    return list(reader)


class StrudelEvaluator(Evaluator):

    def __init__(self,model_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file = model_file


    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print(f"Results file already exists in {self.results_file}, skipping processing")
            return

        ls = LStrudel(n_jobs=self.n_workers)
        train_files = []
        for t in self.train_datasets:
            train_files += [
                f"{self.data_dir}/{t}/{f}"
                for f in os.listdir(os.path.join(self.data_dir, t))
            ]
        train_data = pqdm(
            train_files, load_data, n_jobs=self.n_workers, desc="Loading traning files"
        )

        train_annotation_file = self.data_dir + "train_dev_annotations.jsonl"
        train_annotation_dicts = [
            json.loads(r) for r in open(train_annotation_file, "r").read().splitlines()
        ]
        train_annotations = {d["filename"]: d["line_annotations"] for d in train_annotation_dicts}

        args = [
            {
                "filename": os.path.basename(f),
                "table_array": train_data[idx],
                "line_labels": train_annotations[os.path.basename(f)],
            }
            for idx, f in enumerate(train_files)
        ]
        train_features = pqdm(args[:self.subset], ls.create_line_feature_vector,
                              argument_type="kwargs",
                              n_jobs=self.n_workers, desc="Creating traning features")


        if len([x for x in train_features if type(x) != type(pd.DataFrame())]):
            pdb.set_trace()

        train_df = pd.concat(train_features)
        train_df = train_df[train_df["label"] != "empty"].reset_index(drop=True)

        print("Training model...")
        ls.fit(train_df)

        test_files = [f"{self.data_dir}/{self.test_dataset}/{f}" for f in
            os.listdir(os.path.join(self.data_dir, self.test_dataset))]
        test_data = pqdm(
            test_files, load_data, n_jobs=self.n_workers, desc="Loading testing files"
        )

        if self.test_dataset in ["mendeley","troy"]:
            test_annotation_file = self.data_dir + "test_annotations.jsonl"
        else:
            test_annotation_file = train_annotation_file

        test_annotation_dicts = [
            json.loads(r) for r in open(test_annotation_file, "r").read().splitlines()
        ]
        test_annotations = {d["filename"]: d["line_annotations"] for d in test_annotation_dicts}

        args = [
            {
                "filename": os.path.basename(f),
                "table_array": test_data[idx],
                "line_labels": test_annotations[os.path.basename(f)],
            }
            for idx, f in enumerate(test_files)
        ]
        test_features = pqdm(args[:self.subset], ls.create_line_feature_vector,
                              argument_type="kwargs",
                              n_jobs=self.n_workers, desc="Creating testing features")

        test_df = pd.concat(test_features)
        test_df = test_df[test_df["label"] != "empty"].reset_index(drop=True)

        tmp_results = []
        print("Predicting model...")
        result = ls.predict(test_df)
        tmp_results.append(result)

        results_df = pd.concat(tmp_results, axis=0)
        results_df.to_csv(self.results_file)
        ls.save(self.model_file)

if __name__ == "__main__":
    results = {}
    all_datasets = ["saus", "deex", "cius", "govuk"]
    for dataset in all_datasets + ["mendeley","troy"]:
        print("Validating on dataset", dataset)
        evaluator = StrudelEvaluator(
            data_dir=f"data/line_classification/",
            sys_name="strudel",
            experiment_dir="experiments/finetune_lineclass/",
            train_datasets=[x for x in all_datasets if x != dataset],
            test_dataset=dataset,
            skip_processing=True,
            subset=None,
            n_workers=100,
            model_file=f"weights/finetune_lineclass/strudel_val_{dataset}.pkl",
        )
        evaluator.evaluate()
        res = evaluator.print_results()
        data_f1, header_f1, metadata_f1, group_f1, derived_f1, notes_f1, accuracy, macro_avg = res

        results.update({f"{dataset}_data_f1": data_f1,
                        f"{dataset}_header_f1": header_f1,
                        f"{dataset}_metadata_f1": metadata_f1,
                        f"{dataset}_group_f1": group_f1,
                        f"{dataset}_derived_f1": derived_f1,
                        f"{dataset}_notes_f1": notes_f1,
                        f"{dataset}_accuracy": accuracy,
                        f"{dataset}_macro_avg": macro_avg})
    pd.DataFrame.from_dict(results, orient="index", columns=["strudel"]).to_csv("plots/results/strudel_results.csv", index_label="measure")