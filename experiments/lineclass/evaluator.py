import os
from pathlib import Path
from typing import List, Dict
import numpy as np 
import pandas as pd
from pqdm.processes import pqdm
from sklearn import metrics
import pdb

from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        data_dir: str,
        sys_name: str,
        experiment_dir: str,
        train_datasets: List[str] = [],
        test_dataset: str = "",
        n_repetitions: int = 1,
        n_workers=64,
        subset: int = None,
        skip_processing=False,
    ):
        self.n_repetitions = n_repetitions
        self.subset = subset
        self.data_dir = data_dir
        self.train_datasets = train_datasets
        self.test_dataset = test_dataset

        self.skip_processing = skip_processing
        self.n_workers = n_workers
        self.results_file = experiment_dir + f"/{sys_name}_{test_dataset}_results.csv"

    def process_file(self, in_filepath) -> Dict[str, str, str, float]:
        """

        :param in_filepath: a string containing the full path of the file to process
        :return a dictionary containing the keys "predicted_delimiter", "predicted_quotechar", "predicted_escapechar", "prediction_time"
        """
        raise NotImplementedError

    def print_results(self):
        results_df = pd.read_csv(self.results_file).fillna("").set_index("filename")
        results_df = results_df[results_df["label"]!="empty"]

        (
            data_f1,
            header_f1,
            metadata_f1,
            group_f1,
            derived_f1,
            notes_f1,
        ) = metrics.f1_score(
            results_df["label"],
            results_df["predicted"],
            average=None,
            labels=["data", "header", "metadata", "group", "derived", "notes"],
        )
        accuracy = metrics.accuracy_score(results_df["label"], results_df["predicted"])
        macro_avg = np.mean([metadata_f1, header_f1, group_f1, data_f1, derived_f1, notes_f1])
        print("Results:")
        print("Metadata F1:", metadata_f1)
        print("Header F1:", header_f1)
        print("Group F1:", group_f1)
        print("Data F1:", data_f1)
        print("Derived F1:", derived_f1)
        print("Notes F1:", notes_f1)
        print("Accuracy:", accuracy)
        print("\nMacro avg", macro_avg, "\n")

        return data_f1, header_f1, metadata_f1, group_f1, derived_f1, notes_f1, accuracy, macro_avg

    def process_wrapper(self, in_filepath):
        try:
            return self.process_file(in_filepath)
        except Exception as e:
            print("Error processing file", in_filepath, " : ", str(e))
            return in_filepath + " : " + str(e)

    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print("Skipping processing, results file already exists")
        else:
            Path(self.results_file).parent.mkdir(parents=True, exist_ok=True)
            args = [{"in_filepath": self.in_dir + "/" + f} for f in sorted(self.files)]
            if self.n_workers > 1:
                res = pqdm(
                    args,
                    self.process_wrapper,
                    n_jobs=self.n_workers,
                    argument_type="kwargs",
                    desc="Processing files",
                )
            else:
                res = [
                    self.process_wrapper(**arg)
                    for arg in tqdm(args, desc="Processing files")
                ]

            if len([x for x in res if type(x) != dict]):
                errors = [str(x) for x in res if type(x) != dict]
                print("Error in processing file \n", "\n".join(set(errors)))
                pdb.set_trace()

            results_df = pd.DataFrame(res).fillna("").set_index("filename")
            results_df.to_csv(self.results_file)
