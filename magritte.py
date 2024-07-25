import warnings
import logging
logging.captureWarnings(True)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

warnings.filterwarnings("ignore", "torchtext", UserWarning)
warnings.filterwarnings("ignore", "Some weights of the model")
import torchtext
from csv_embedder.magritte_finetune_rowclass.model import MagritteFinetuneRowClassification


from csv_embedder.reca_columntype.model import KREL; 
torchtext.disable_torchtext_deprecation_warning()
from csv_embedder.magritte_base.model import MagritteBase
from csv_embedder.magritte_finetune_dialect.model import MagritteFinetuneDialectDetection

import os
import json
import pandas as pd

def load_model(config_path, model_class):
    config = json.load(open(config_path))
    model = model_class(**config["model"])
    model.load_weights(config["model"]["save_path"])
    return model

class MaGRiTTE():

    dialect_model = load_model("configs_json/dialect.json", MagritteFinetuneDialectDetection)
    rowclass_model = load_model("configs_json/rowclass.json", MagritteFinetuneRowClassification)
    coltype_model = load_model("configs_json/columntype.json", KREL)

    @classmethod
    def extract_table(cls, filepath):
        prediction = cls.dialect_model.predict(filepath)
        if prediction["escapechar"] == '"':
            escape = {"escapechar": None, "doublequote": True}
        else:
            escape = {"escapechar": prediction["escapechar"], "doublequote": False}
        table = pd.read_csv(filepath,
                            delimiter=prediction["delimiter"], 
                            quotechar=prediction["quotechar"], 
                            **escape,
                            header=None,
                            engine="python",
                            on_bad_lines="skip")
        table = table.dropna(how= "all", axis=1)
        return table

    @classmethod
    def delete_metadata(cls, table: pd.DataFrame):
        filerows = table.to_csv(index=False, header=False)
        rowclasses = cls.rowclass_model.predict(filerows)
        try:
            header_idx = rowclasses.index("header")
            table = table.rename(columns=table.iloc[header_idx])
        except ValueError: # no header row was detected.
            table = table.rename(columns={col:None for col in table.columns})
        metadata = ["header", "group", "notes", "metadata", "empty"]
        max_row = len(table)
        delete_idx = [idx for idx, rowclass in enumerate(rowclasses) if (rowclass in metadata) and (idx < max_row)]
        return table.drop(delete_idx).reset_index(drop=True)

    @classmethod
    def disambiguate_column_headers(cls, table: pd.DataFrame):
        old_table = table.copy()
        columns = [" ".join(map(str,table[x].values)) for x in table.columns]
        headers = cls.coltype_model.predict(columns)
        # If column names are NaN, replace them with the detected headers
        counts = {}
        new_header = []
        for idx, col in enumerate(table.columns):
            if pd.isna(col):
                new_col = headers[idx]
                cur_count = counts.get(headers[idx], 0)
                if cur_count > 0:
                    new_col += f"_{cur_count}"
                counts[headers[idx]] = cur_count + 1
                new_header.append(new_col)
            else:
                new_header.append(col)
        
        if len(new_header) != len(list(set(new_header))):
            breakpoint()

        table.columns = new_header
        return table