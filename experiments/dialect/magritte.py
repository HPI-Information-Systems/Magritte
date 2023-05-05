import csv
import sys

sys.path.append(".")
sys.path.append("..")

import json
import _jsonnet # type: ignore
import lightning.pytorch as pl # type: ignore
import pandas as pd
from torch.utils.data import DataLoader # type: ignore
from torchtext.vocab import vocab as build_vocab # type: ignore
from sklearn import metrics

from csv_embedder import PatternTokenizer
from csv_embedder.magritte_finetune_dialect.dataset import DialectDataset
from csv_embedder.magritte_finetune_dialect.model import MagritteFinetuneDialectDetection

from evaluator import Evaluator

import logging
logging.getLogger('lightning').setLevel(0)

class MagritteEvaluator(Evaluator):

    def __init__(self,
                 config_path="configs/dialect.jsonnet",
                 weights_path="",
                 global_max=True,
                 cuda_device=-1,
                 batch_size = 16,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        self.config_path = config_path
        self.weights_path = weights_path
        self.global_max = global_max
        self.cuda_device = cuda_device

        config = _jsonnet.evaluate_file(self.config_path,
                                        ext_vars={"max_len": "128",
                                                  "encoding_dim": "128"})
        config = json.loads(config)
        self.batch_size = batch_size
        config["trainer"]["devices"] = [self.cuda_device]

        self.max_rows = config["data_module"]["max_rows"]
        self.max_len = config["data_module"]["max_len"]
        config["data_module"]["batch_size"] = self.batch_size

        tokens = open(config["vocabulary"]["directory"] + "/tokens.txt").read().splitlines()
        ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
        self.token_vocab = build_vocab(ordered_tokens)
        self.token_vocab.set_default_index(self.token_vocab["[UNK]"])

        dialect_classes = open(config["vocabulary"]["directory"] + "/dialect_labels.txt").read().splitlines()
        ordered_classes = {c: len(dialect_classes) - i for i, c in enumerate(dialect_classes)}
        self.label_vocab = build_vocab(ordered_classes)

        self.magritte = MagritteFinetuneDialectDetection(token_vocab=self.token_vocab,
                                                         label_vocab=self.label_vocab,
                                                         **config["model"])
        self.magritte.load_weights(weights_path)
        self.magritte.eval()

        self.trainer = pl.Trainer(**config["trainer"])

    def evaluate(self, *args, **kwargs):
        df = pd.read_csv(self.dialect_file).fillna("")
        df = df[df['filename'].isin(self.files)]

        dataset = DialectDataset(df,
                                 data_path=self.data_dir,
                                 token_vocab=self.token_vocab,
                                 label_vocab=self.label_vocab,
                                 max_rows=self.max_rows,
                                 max_len=self.max_len,
                                 tokenizer=PatternTokenizer(),
                                 for_prediction=True,)

        dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        batches = self.trainer.predict(self.magritte, dl)
        lookup_vocab = {i: k for i, k in enumerate(self.token_vocab.get_itos())}
        unk_idx = self.token_vocab(["[UNK]"])[0]

        lookup_vocab[unk_idx] = ""

        predicted_delimiter = []
        predicted_quotechar = []
        predicted_escapechar = []
        target_delimiter = []
        target_quotechar = []
        target_escapechar = []

        for batch in batches:
            y, target = batch
            predicted_delimiter += [lookup_vocab[i] for i in y["predicted_delimiter"].cpu().numpy()]
            predicted_quotechar += [lookup_vocab[i] for i in y["predicted_quotechar"].cpu().numpy()]
            predicted_escapechar += [lookup_vocab[i] for i in y["predicted_escapechar"].cpu().numpy()]
            if type(target["target_delimiter"]) == type(y["predicted_delimiter"]):
                target_delimiter += [lookup_vocab[i] for i in target["target_delimiter"].cpu().numpy()]
                target_quotechar += [lookup_vocab[i] for i in target["target_quotechar"].cpu().numpy()]
                target_escapechar += [lookup_vocab[i] for i in target["target_escapechar"].cpu().numpy()]
            else:
                target_delimiter.extend(target["target_delimiter"])
                target_quotechar.extend(target["target_quotechar"])
                target_escapechar.extend(target["target_escapechar"])

        results_df = pd.DataFrame({
            "filename": df["filename"],
            "predicted_delimiter": predicted_delimiter,
            "predicted_quotechar": predicted_quotechar,
            "predicted_escapechar": predicted_escapechar,
            "target_delimiter": target_delimiter,
            "target_quotechar": target_quotechar,
            "target_escapechar": target_escapechar,
            "prediction_time": 0,
        }).fillna("").set_index("filename")

        print("Evaluated on ", len(results_df), "files")
        print("Saving results to ", self.results_file)
        results_df.to_csv(self.results_file, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    weights_path = "weights/magritte_dialect.pth"
    print("SUT: Magritte ")
    evaluator = MagritteEvaluator(data_dir="data/dialect_detection/dev_augmented/",
                                    sys_name="magritte",
                                    experiment_dir="experiments/finetune_dialect/",
                                    dataset="dev_augmented",
                                    subset=None,
                                    n_workers=100,
                                    skip_processing=False,
                                    augmented=True,
                                    original=True,
                                    weights_path=weights_path,
                                    global_max=True,
                                    cuda_device=0,
                                    batch_size=80,
                                    )
    evaluator.evaluate()
    evaluator.print_results()

    print("Test set:")
    test_evaluator = MagritteEvaluator(data_dir = "data/dialect_detection/test/",
                                    sys_name="magritte",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "test",
                                    augmented=False,
                                    original=True,
                                    subset=None,
                                    n_workers=100,
                                    skip_processing=False,
                                    weights_path=weights_path,
                                    global_max=True,
                                    cuda_device=0,
                                    batch_size=80,)
    test_evaluator.evaluate()
    test_evaluator.print_results()
 
    print("Difficult set:")
    test_evaluator = MagritteEvaluator(data_dir = "data/dialect_detection/difficult/",
                                    sys_name="magritte",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "difficult",
                                    augmented=False,
                                    original=True,
                                    subset=None,
                                    n_workers=100,
                                    skip_processing=False,
                                    weights_path=weights_path,
                                    global_max=True,
                                    cuda_device=0,
                                    batch_size=80,)
    test_evaluator.evaluate()
    test_evaluator.print_results()