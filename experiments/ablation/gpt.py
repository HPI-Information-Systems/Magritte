import time
import pdb
import os
from openai import OpenAI
import tiktoken
import time
import pandas as pd
import json
from tqdm import tqdm
import regex as re
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support

def getJsonFromAnswer(answer: str):
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    if not answer.strip().startswith("{"):
        # Find the start index of the actual JSON string
        # assuming the prefix is followed by the JSON object/array
        start_index = answer.find("{") if "{" in answer else answer.find("[")
        if start_index != -1:
            # Remove the prefix and any leading characters before the JSON starts
            answer = answer[start_index:]

    if not answer.strip().endswith("}"):
        # Find the end index of the actual JSON string
        # assuming the suffix is preceded by the JSON object/array
        end_index = answer.rfind("}") if "}" in answer else answer.rfind("]")
        if end_index != -1:
            # Remove the suffix and any trailing characters after the JSON ends
            answer = answer[: end_index + 1]

    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace(r"\_", "_")

    # Handle comments in the JSON response. Use regex from // until end of line
    answer = re.sub(r"\/\/.*$", "", answer, flags=re.MULTILINE)
    return json.loads(answer)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_TOKENS = 4096
ANSWER_TOKENS = 1024

# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("text-davinci-003")

INSTRUCTION = """The following text is a tabular data file (in CSV-like format) which encodes a table.
For this file, I want to know 1. the number of rows, 2. the number of columns, 3. the number of header lines, 4. the number of preamble lines (if any), 5. the number of footnote lines (if any).
Provide the output as a JSON object with the following keys: 'n_rows', 'n_cols', 'n_header_lines', 'n_preamble_lines', 'n_footnote_lines'. Each key should have an integer value."""

SEED = 42

class GPTModel():

    def __init__(self, 
                 model= 'gpt-3.5-turbo-1106', 
                 data_dir = "/data/survey/csv",
                 dataset = "/data/survey/test.csv",
                 outpath = "results/ablation_regression/gpt35/",
                 subset = 5,
                 *args, **kwargs):
        
        self.client = OpenAI(api_key = OPENAI_API_KEY)
        self.model = model
        if model == 'gpt-4o':
            self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-05-13")
        else:
            self.encoding = tiktoken.encoding_for_model(model)
        self.data_dir = data_dir
        self.dataset = dataset
        self.outpath = outpath
        self.subset = subset
        self.batch_file = os.path.join(self.outpath, "test_batch.jsonl")
        self.batch_status_file = os.path.join(self.outpath, "batch_status.json")
        self.results_file = os.path.join(self.outpath, "results.json")

    def create_batch(self):
        dicts = []
        dataset = pd.read_csv(self.dataset)
        filenames = dataset['filename'][:self.subset]
        for idx,f in tqdm(enumerate(filenames), total=len(filenames), desc="Creating batch"):
            in_filepath = os.path.join(self.data_dir, f)
            try:
                with open(in_filepath, newline='', encoding="latin-1") as in_csvfile:
                    file_text = in_csvfile.read(1024*1024)
            except Exception as e:
                with open(in_filepath, newline='', encoding="utf8") as in_csvfile:
                    file_text = in_csvfile.read(1024*1024)

            num_prompt_tokens = len(self.encoding.encode(INSTRUCTION))
            max_file_tokens = MAX_TOKENS-num_prompt_tokens
            file_tokens = self.encoding.encode(file_text)
            if len(file_tokens) > max_file_tokens:
                file_text = self.encoding.decode(file_tokens[:max_file_tokens])

            # res = self.process_file(in_filepath)
            # with open(self.outpath + f + "_results.json", "w") as f:
            #     f.write(json.dumps(res))

            request= {"custom_id": f"request-{idx}", 
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": {"model": self.model, 
                             "messages": 
                             [{"role": "system", "content": INSTRUCTION},
                              {"role": "user", "content": file_text}],
                              "max_tokens": MAX_TOKENS}}
            dicts.append(request)
        
        with open(self.batch_file, "w") as f:
            for d in dicts:
                f.write(json.dumps(d) + "\n")

    def send_batch(self):
        batch_input_file = self.client.files.create(
            file=open(self.batch_file, "rb"),
            purpose="batch"
            )

        batch_input_file_id = batch_input_file.id

        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "benchmark suc test job"
            }
        )

        self.update_batch_status(batch_obj)

    def update_batch_status(self, batch_obj):
        with open(self.batch_status_file, "w") as f:
            f.write(json.dumps(batch_obj.to_dict()))

    def get_batch_status(self):
        with open(self.batch_status_file, "r") as f:
            batch_status = json.loads(f.read())
        batch_id = batch_status["id"]
        batch_status = self.client.batches.retrieve(batch_id)
        print(batch_status.to_dict())
        self.update_batch_status(batch_status)
        return batch_status

    def get_file_response(self):
        with open(self.batch_status_file, "r") as f:
            batch_status = json.loads(f.read())
        file_id = batch_status["output_file_id"]

        file_response = self.client.files.content(file_id)
        with open(self.results_file, "w") as f:
            f.write(file_response.text)

    def evaluate_results(self):
        with open(self.results_file, "r") as f:
            resultdicts = f.read().splitlines()

        dataset = pd.read_csv(self.dataset)
        filenames = dataset['filename'][:self.subset]

        predicted_df = []
        for idx, result in enumerate(resultdicts):
            dct = json.loads(result)
            response = dct["response"]["body"]["choices"][0]["message"]["content"]
            try:
                resdict = getJsonFromAnswer(response)
                assert 'n_rows' in resdict
                assert 'n_cols' in resdict
                assert 'n_header_lines' in resdict
                assert 'n_preamble_lines' in resdict
                assert 'n_footnote_lines' in resdict
            except:
                print("Error in JSON response:", response)
                breakpoint()
            predicted_df.append({"filename": filenames[idx],
                               'n_rows': resdict['n_rows'],
                               'n_cols': resdict['n_cols'],
                               'n_header_lines': resdict['n_header_lines'],
                               'n_preamble_lines': resdict['n_preamble_lines'],
                               'n_footnote_lines': resdict['n_footnote_lines']})
        predicted_df = pd.DataFrame(predicted_df)
        predicted_df.to_csv(self.outpath + "results.csv", index=False)
        target_df = pd.read_csv(self.dataset)

        file_index = target_df['filename'].values
        predicted_df.set_index('filename', inplace=True)
        target_df.set_index('filename', inplace=True)
        predicted_df = predicted_df.reindex(target_df.index).fillna(0)
        
        target_df["n_preamble_lines"] = target_df["table_preamble_rows"]
        target_df["n_footnote_lines"] = target_df["table_footnote_rows"]
        # for row in target_df.iterrows():
            # if row[1]["table_no_header"] == 1:
                # target_df.at[row[0], "n_header_lines"] = 0
            # elif row[1]["table_multirow_header"] == 0:
                # target_df.at[row[0], "n_header_lines"] = 1
            # else:
                # target_df.at[row[0], "n_header_lines"] = row[1]["table_multirow_header"]
        for row in predicted_df.iterrows():
            if row[1]["n_header_lines"] == 1 or row[1]["n_header_lines"] == 0:
                predicted_df.at[row[0], "table_multirow_header"] = 0
            else:
                predicted_df.at[row[0], "table_multirow_header"] = 1

            if row[1]["n_preamble_lines"] == 1 or row[1]["n_preamble_lines"] == 0:
                predicted_df.at[row[0], "table_preamble_rows"] = 0
            else:
                predicted_df.at[row[0], "table_preamble_rows"] = 1

            if row[1]["n_footnote_lines"] == 1 or row[1]["n_footnote_lines"] == 0:
                predicted_df.at[row[0], "table_footnote_rows"] = 0
            else:
                predicted_df.at[row[0], "table_footnote_rows"] = 1


        for col in ['n_rows', 
                    'n_cols', 
                    'table_multirow_header', 
                    'table_preamble_rows', 
                    'table_footnote_rows']:
            pre = predicted_df[col].values
            target = target_df[col].values

            if col in ('n_rows','n_cols'):
                norm = max(max(pre),max(target))
            else:
                norm = 1
            # elif col in ('n_header_lines','n_preamble_lines','n_footnote_lines'):
                # norm = 128       
            pre = pre/norm
            target = target/norm
            mse = mean_squared_error(target, pre)
            mae = mean_absolute_error(target, pre)
            if col in ('n_rows','n_cols'):
                p, r, f1, = -1,-1,-1
            else:
                p,r,f1, support = precision_recall_fscore_support(target, pre, average='binary')

            
            print(mse)
            print(mae)
            print(p)
            print(r)
            print(f1)
            # print(f"Mean Squared Error for {col}: {mse}")
            # print(f"Mean Absolute Error for {col}: {mae}")
            # print(f"Precision for {col}: {p}")
            # print(f"Recall for {col}: {r}")
            # print(f"F1 for {col}: {f1}")
            # print() 

if __name__ == "__main__":

    print("Test GPT 3.5")
    gpt3 = GPTModel(
                              model= 'gpt-3.5-turbo-1106',
                              data_dir = "/data/survey/csv", 
                              dataset = "/data/survey/test.csv", 
                              outpath = "results/ablation_regression/gpt35/",
                              subset = None,
                              n_workers=0)
    if not os.path.exists(gpt3.batch_file):
        gpt3.create_batch()

    if not os.path.exists(gpt3.batch_status_file):
        gpt3.send_batch()

    batch_status = gpt3.get_batch_status()
    if batch_status.to_dict()["status"] == "completed":
        gpt3.get_file_response()
        gpt3.evaluate_results()
    gpt3.evaluate_results()
    
    print("Test GPT 4:")
    gpt4 = GPTModel(model= 'gpt-4o',
                    data_dir = "/data/survey/csv", 
                    dataset = "/data/survey/test.csv", 
                    outpath = "results/ablation_regression/gpt4/",
                    subset = None,
                    n_workers=0)

    if not os.path.exists(gpt4.batch_file):
        gpt4.create_batch()

    if not os.path.exists(gpt4.batch_status_file):
        gpt4.send_batch()

    batch_status = gpt4.get_batch_status()
    if batch_status.to_dict()["status"] == "completed":
        gpt4.get_file_response()
    gpt4.evaluate_results()