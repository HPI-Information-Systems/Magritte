import pdb
import os
import openai
import tiktoken
import time
import pandas as pd


from evaluator import Evaluator

OPENAI_API_KEY = "your-token-here"
MAX_TOKENS = 4096
ANSWER_TOKENS = 32

openai.api_key = OPENAI_API_KEY
openai.Model.list()

enc = tiktoken.get_encoding("p50k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("text-davinci-003")

class GPTEvaluator(Evaluator):

    def process_file(self, in_filepath):
        start = time.process_time()
        prompt_text = "Identify the delimiter, quotation and escape characters of the following file.\n File: "
        end_text = "\nDelimiter, quotation, escape: "
        num_tokens = len(enc.encode(prompt_text))+len(enc.encode(end_text))
        try:
            with open(in_filepath, newline='', encoding="utf8") as in_csvfile:
                file_text = in_csvfile.read(8*1024*1024)
        except Exception as e:
            with open(in_filepath, newline='', encoding="latin-1") as in_csvfile:
                file_text = in_csvfile.read(8*1024*1024)

        max_file_tokens = MAX_TOKENS-num_tokens-ANSWER_TOKENS

        file_tokens = enc.encode(file_text)
        if len(file_tokens) + num_tokens+ ANSWER_TOKENS > MAX_TOKENS:
            file_text = enc.decode(file_tokens[:max_file_tokens])

        prompt_text += file_text + end_text

        response = openai.Completion.create(
            model='text-davinci-003', 
            prompt=prompt_text,
            temperature=0,
            max_tokens=ANSWER_TOKENS,
            top_p=1,
        )
        response = response['choices'][0]['text']
        

        end = time.process_time()-start
        return {"filename": os.path.basename(in_filepath),
                "response" : response,
                "prediction_time": end}


if __name__ == "__main__":

    print("Overall set:")
    test_evaluator = GPTEvaluator(data_dir = "data/dialect_detection/overall/",
                                     sys_name="gpt",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "overall",
                                     subset=100,
                                     n_workers=0,
                                     skip_processing=True,
                                     augmented=False,
                                    original=True)
    test_evaluator.evaluate()
    test_evaluator.print_results()