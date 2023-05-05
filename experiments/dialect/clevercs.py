import os
import time

import clevercsv

from evaluator import Evaluator

class CleverEvaluator(Evaluator):

    def process_file(self, in_filepath):
        start = time.process_time()
        try:
            with open(in_filepath, newline='', encoding="utf8") as in_csvfile:
                dialect = clevercsv.Sniffer().sniff(in_csvfile.read(8*1024*1024))
        except Exception as e:
            try:
                with open(in_filepath, newline='', encoding="latin-1") as in_csvfile:
                    dialect = clevercsv.Sniffer().sniff(in_csvfile.read(8*1024*1024))
            except Exception as e:
                print(f'Error processing file {in_filepath}: {e}')
                dialect = None

        end = time.process_time()-start
        if dialect is not None:
            return {"filename": os.path.basename(in_filepath),
                    "predicted_delimiter": dialect.delimiter,
                    "predicted_quotechar": dialect.quotechar,
                    "predicted_escapechar": dialect.escapechar
                    , "prediction_time": end}
        else:
            return {"filename": os.path.basename(in_filepath),
                    "predicted_delimiter": "[UNK]",
                    "predicted_quotechar": "[UNK]",
                    "predicted_escapechar": "[UNK]",
                    "prediction_time": end}


if __name__ == "__main__":

    print("SUT: CleverCSV")
    evaluator = CleverEvaluator(data_dir = "data/dialect_detection/dev_augmented/",
                                sys_name = "clevercs",
                                experiment_dir = "experiments/finetune_dialect/",
                                dataset = "dev_augmented",
                                skip_processing = False,
                                subset=None,
                                n_workers=60,
                                augmented=True,
                                original=True,
                                )
    evaluator.evaluate()
    evaluator.print_results()

    print("Test set:")
    test_evaluator = CleverEvaluator(data_dir = "data/dialect_detection/test/",
                                     sys_name="clevercs",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "test",
                                     subset=None,
                                     n_workers=60,
                                     augmented=False,
                                    original=True)
    test_evaluator.evaluate()
    test_evaluator.print_results()

    print("Difficult set:")
    test_evaluator = CleverEvaluator(data_dir = "data/dialect_detection/difficult/",
                                     sys_name="clevercs",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "difficult",
                                     subset=None,
                                     n_workers=60,
                                     augmented=False,
                                    original=True)
    test_evaluator.evaluate()
    test_evaluator.print_results()

    print("Overall set:")
    test_evaluator = CleverEvaluator(data_dir = "data/dialect_detection/overall/",
                                     sys_name="clevercs",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "overall",
                                     subset=None,
                                     n_workers=60,
                                     augmented=False,
                                    original=True)
    test_evaluator.evaluate()
    test_evaluator.print_results()