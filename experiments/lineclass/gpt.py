import re
import json
import os
import pdb
import tqdm
import os.path
import numpy as np
from sklearn import metrics
import csv
import pandas as pd
import sys
import openai
import tiktoken

from evaluator import Evaluator

OPENAI_API_KEY = "your-api-key"
MAX_TOKENS = 4096
ANSWER_TOKENS = 100
openai.api_key = OPENAI_API_KEY

enc = tiktoken.get_encoding("p50k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("text-davinci-003")


def load_data(file_path):
    reader = csv.reader(open(file_path, "r"))
    return list(reader)


class GPTEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_file = self.results_file.replace(".csv", "_response.csv")


    def parse_response(self, clas,row):
        f = row["filename"]
        response = row["response"]
        parse = re.split(f'\n{clas}:\s(.*?)\n',response)
        if len(parse)>1:
            if "None" in parse[1]:
                return []
            else:
                try:
                    return list(map(int,parse[1].split(",")))
                except Exception as e:
                    filepath = f"{self.data_dir}/{self.test_dataset}/{f}"
                    with open(filepath, newline='', encoding="utf8") as in_csvfile:
                        lines = in_csvfile.read(8*1024*1024).split("\n")

                    for idx,l in enumerate(lines):
                        if parse[1].replace(" ","") in l:
                            return [idx]
                    return []
        else:
            return []


    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print(f"Results file already exists in {self.results_file}, skipping processing")
            return

        if not (self.skip_processing and os.path.exists(self.response_file)):

            preamble_text = """Header lines represent the column names of tables; data lines represent records; group lines organize tables into sub-tables and are the header for a given group; derived lines contain the result of some operation on data lines; metadata and note lines contain metadata information respectively before and after tables. In the following CSV file, identify what lines are header, group header, metadata, note, and derived. File:\n"""
            end_text = "\nLists of indices of header, data, group header, metadata, note, derived:"
            num_tokens = len(enc.encode(preamble_text))+len(enc.encode(end_text))
            
            test_files = [f"{self.data_dir}/{self.test_dataset}/{f}" for f in
                os.listdir(os.path.join(self.data_dir, self.test_dataset))]
            
            max_file_tokens = MAX_TOKENS-num_tokens-ANSWER_TOKENS
            tmp_results = []
            for filepath in tqdm.tqdm(test_files[:self.subset], desc="Querying GPT-3.5"):
                try:
                    with open(filepath, newline='', encoding="utf8") as in_csvfile:
                        file_text = in_csvfile.read(8*1024*1024)
                except Exception as e:
                    with open(filepath, newline='', encoding="latin-1") as in_csvfile:
                        file_text = in_csvfile.read(8*1024*1024)

                file_tokens = enc.encode(file_text)
                if len(file_tokens) > max_file_tokens:
                    file_text = enc.decode(file_tokens[:max_file_tokens])

                prompt_text = preamble_text + file_text + end_text
                response = openai.Completion.create(
                    model='text-davinci-003', 
                    prompt=prompt_text,
                    temperature=0,
                    max_tokens=ANSWER_TOKENS,
                    top_p=1,
                )
                response = response['choices'][0]['text']
                tmp_results.append([{"filename":os.path.basename(filepath), 
                                    "response":response}])

            response_df = pd.DataFrame.from_dict(tmp_results)
            response_df.to_csv(self.response_file)

        response_df = pd.read_csv(self.response_file)

        response_df["header"] = response_df.apply(lambda x: self.parse_response("Header", x), axis=1)
        response_df["group"] = response_df.apply(lambda x: self.parse_response("Group",x), axis=1)
        response_df["metadata"] = response_df.apply(lambda x: self.parse_response("Metadata",x), axis=1)
        response_df["note"] = response_df.apply(lambda x: self.parse_response("Note",x), axis=1)
        response_df["derived"] = response_df.apply(lambda x: self.parse_response("Derived",x), axis=1)
        response_df["data"] = response_df.apply(lambda x: self.parse_response("Data",x), axis=1)

        if self.test_dataset in ["mendeley","troy"]:
            test_annotation_file = self.data_dir + "test_annotations.jsonl"
        else:
            test_annotation_file = self.data_dir + "train_dev_annotations.jsonl"

        test_annotation_dicts = [
            json.loads(r) for r in open(test_annotation_file, "r").read().splitlines()
        ]
        test_annotations = {d["filename"]: d["line_annotations"] for d in test_annotation_dicts}

        results = []
        for idx,row in response_df.iterrows():
            file_rows = load_data(f"{self.data_dir}/{self.test_dataset}/{row['filename']}")
            for idx in range(len(file_rows)):
                if idx in row["header"]:
                    predicted = "header"
                elif idx in row["group"]:
                    predicted = "group"
                elif idx in row["metadata"]:
                    predicted = "metadata"
                elif idx in row["note"]:
                    predicted = "note"
                elif idx in row["derived"]:
                    predicted = "derived"
                else:
                    predicted = "data"

                results.append({"filename":row["filename"], "row":idx, "predicted":predicted, "label":test_annotations[row["filename"]][idx]})


        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(self.results_file)


if __name__ == "__main__":
    results = {}
    all_datasets = ["saus", "deex", "cius", "govuk"]
    all_df = pd.DataFrame()
    for dataset in all_datasets + ["mendeley","troy"]:
        print("Validating on dataset", dataset)
        evaluator = GPTEvaluator(
            data_dir=f"data/line_classification/",
            sys_name="gpt",
            experiment_dir="experiments/finetune_lineclass/",
            test_dataset=dataset,
            skip_processing=True,
            subset=20,
            n_workers=0,
        )
        evaluator.evaluate()
        results_df = pd.read_csv(evaluator.results_file)
        all_df = pd.concat([all_df, results_df], axis=0)

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

    (
        data_f1,
        header_f1,
        metadata_f1,
        group_f1,
        derived_f1,
        notes_f1,
    ) = metrics.f1_score(
        all_df["label"],
        all_df["predicted"],
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

    pd.DataFrame.from_dict(results, orient="index", columns=["strudel"]).to_csv("../results/gpt_results.csv", index_label="measure")
