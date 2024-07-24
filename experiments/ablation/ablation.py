import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from pathlib import Path
import pickle
import torchtext
from sklearn.metrics import PrecisionRecallDisplay

# torchtext.disable_torchtext_deprecation_warning()
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Some weights of the model checkpoint*")
import _jsonnet

import json
import os
import sys
import gc

sys.path.append(".")
sys.path.append("../../")

import pandas as pd
from tqdm import tqdm as tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import time

from torchtext.vocab import vocab as build_vocab
from experiments.ablation.model import AblationModel, EmbeddingsModel
from experiments.ablation.dataset import EmbeddingsDataset
from experiments.ablation.factory import EmbeddingsFactory, Turl_embedder

seed_everything(42, workers=True)
regressor_log = {
    "embedding_model": [],
    "regression_feature": [],
    "train_embedding_time (sec)": [],
    "dev_embedding_time (sec)": [],
    "regressor_training_time (sec)": [],
}
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# SUC Benchmark:
# Table partition: line of the first row of a table,
# line of the last row of a table (excluding preamble and footnotes)
# Table size detection: number of rows,
# number of columns
# Hierarchy detection: detecting number of header rows
def drop_rows_overtimed(dataset: pd.DataFrame, to_drop: list) -> pd.DataFrame:
    for d in to_drop:
        dataset = dataset.drop(dataset[dataset['filename'] == d].index)
    return dataset

def train_regressors(
    model: EmbeddingsModel,
    train_dataset: pd.DataFrame,
    dev_dataset: pd.DataFrame,
    checkpoints_directory: str,
    config: dict,
    use_cache: bool = True,
) -> dict:
    """Given a training and a dev datasets, train a regressor with the specified configuration.

    Args:
        train_dataset (pd.DataFrame): dataset that links the training table names with their labels
        dev_dataset (pd.DataFrame): dataset that links the validation table names with their labels
        checkpoints_directory (str): path to the directory where to save the checkpoints of the models trained, a new folder named with the current time will be created will be created inside it
        config (dict): configuration dictionary that contains the model name and the feature to train the regressor on.

    Returns:
        dict: 2 layered dictionary that contains the trained regressors.
        It contains an entry for every model, ths entries contain each one a dictionary with an entry for each feature, here are stored the regressors.
        Example of usage: trained_regressor = regressors['magritte']['n_cols']
    """
    regressor_path = f"{checkpoints_directory}/regressor.pth"
    model_name = config["model_name"]
    feature = config["feature"]

    train_embeddings = model.get_embeddings(
        filenames=train_dataset["filename"], model_name=model_name, dataset_name="train"
    )

    if use_cache and os.path.exists(regressor_path):
        regressor = AblationModel(embedding_size=train_embeddings.size(1), 
                                  feature=config["feature"],
                                  optimizer_lr=config["optimizer_lr"])
        print("\tLoading regressor from ", regressor_path)
        regressor.load_state_dict(torch.load(regressor_path))
        time_logs = pickle.load(open(f"{checkpoints_directory}/time_logs.pkl", "rb"))
        return regressor, time_logs

    if isinstance(model, Turl_embedder):
        to_drop = model.embedding_dictionary['overtime']
        train_dataset = drop_rows_overtimed(train_dataset, to_drop)
        dev_dataset = drop_rows_overtimed(dev_dataset, to_drop)

    start = time.time()
    train_embeddings = model.get_embeddings(
        filenames=train_dataset["filename"], model_name=model_name, dataset_name="train"
    )
    train_emb_time = time.time() - start

    start = time.time()
    dev_embeddings = model.get_embeddings(
        filenames=dev_dataset["filename"], model_name=model_name, dataset_name="dev"
    )
    dev_emb_time = time.time() - start

    time_logs = {
        "embedding_model": model_name,
        "train_embedding_time_sec": train_emb_time,
        "dev_embedding_time_sec": dev_emb_time,
        "regression_feature": feature,
    }

    start = time.time()

    train_labels = train_dataset[feature]
    dev_labels = dev_dataset[feature]

    train_ds = EmbeddingsDataset(train_embeddings, train_labels)
    train_dataloader = DataLoader(train_ds, num_workers=0, batch_size=config["batch_size"])
    dev_ds = EmbeddingsDataset(dev_embeddings, dev_labels)
    dev_dataloader = DataLoader(dev_ds, num_workers=0, batch_size=config["batch_size"])

    pos_weights = 1-(train_labels.sum() / len(train_labels)) # Positive class imbalances
    
    if feature in ('n_cols', 'n_rows'):
        pos_weights = None
    
    regressor = AblationModel(train_embeddings.size(1), 
                              feature=feature, 
                              pos_weights=pos_weights,
                              optimizer_lr=config["optimizer_lr"])

    callbacks = [ModelCheckpoint(**config["callbacks"]["model_checkpoint"])]
    # if config["early_stopping"]:
    callbacks += [EarlyStopping(**config["callbacks"]["early_stopping"])]

    trainer = pl.Trainer(**config["trainer"], callbacks=callbacks)
    trainer.fit(
        model=regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=dev_dataloader,
    )
    torch.save(regressor.state_dict(), regressor_path)
    time_logs["regressor_training_time_sec"] = time.time() - start
    pickle.dump(time_logs, open(f"{checkpoints_directory}/time_logs.pkl", "wb"))
    return regressor, time_logs


def evaluate(
    regressor,
    eval_dataset,
    config,
    model,
    model_name: str,
    feature: str,
    model_feature_name: str
) -> set:
    if isinstance(model, Turl_embedder):
        overtimes = model.embedding_dictionary['overtime']
        zero_indexes = []
        filenames_dataset = eval_dataset['filename']
        for r in range(eval_dataset.shape[0]):
            if filenames_dataset[r] in overtimes:
                zero_indexes.append(r)

    embeddings = model.get_embeddings(
        filenames=eval_dataset["filename"], model_name=model_name, dataset_name="test")
        # filenames=eval_dataset["filename"], model_name=model_name, dataset_name="test")
    labels = eval_dataset[feature]

    regressor.eval()

    eval_ds = EmbeddingsDataset(embeddings, labels)
    assert len(eval_ds)==embeddings.shape[0]
    assert embeddings.shape[0]==len(labels)
    dataloader = DataLoader(eval_ds, num_workers=0, batch_size=256)
    trainer = pl.Trainer(**config["trainer"])
    with torch.no_grad():
        batches = trainer.predict(regressor, dataloader)
        predicted = []
        targets = []
        for batch in batches:
            y, y_true = batch
            predicted += y.squeeze().numpy().tolist()
            targets += y_true.squeeze().numpy().tolist()
    if isinstance(model, Turl_embedder):
        for i in zero_indexes:
            predicted[i] = 0

    predictions_raw = {model_feature_name+'_pred':predicted, model_feature_name+'_true':targets}
    mse = mean_squared_error(targets, predicted)
    mae = mean_absolute_error(targets, predicted)
    
    if feature in ['table_multirow_header', 'table_preamble_rows', 'table_footnote_rows']:
        predicted = [1 if p >= 0 else 0 for p in predicted]
        # targets = [1 if t >= 0.5 else 0 for t in targets]
        precision = precision_score(targets, predicted)
        recall = recall_score(targets, predicted)
        f1 = f1_score(targets, predicted)
    else:
        precision = -1
        recall = -1
        f1 = -1

    dict_out = {
        "mse": mse,
        "mae": mae,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions_raw
    }

    return dict_out


CONFIG_PATH = f"configs/ablation_regression.jsonnet"

if __name__ == "__main__":

    path = "/data/survey/"
    train_dataset = pd.read_csv(path + "train.csv")
    dev_dataset = pd.read_csv(path + "dev.csv")

    ckpt_dir = "/data/magritte_results/ablation_regression/"

    embedding_models = [
        # "turl",
        "magritte-pt-sfp-cnn",
        "magritte-pt-cnn",
        "magritte-pt-sfp",
        "magritte-pt",
        "magritte-wp-sfp-cnn",
        "magritte-wp-cnn",
        "magritte-wp-sfp",
        "magritte-wp",
        "bert_mean",
        "roberta_mean",
    ]
    features = [
        "n_rows",
        "n_cols",
        "table_multirow_header",
        "table_preamble_rows",
        "table_footnote_rows"
    ]

    # if False:
    if True:
        for model in embedding_models:
            for feature in features:
                if "cnn" not in model:
                    continue
                print("Removing previous regressor:", f"{model}_{feature}/")
                try:
                    shutil.rmtree(f"{ckpt_dir}/{model}_{feature}/")
                except:
                    pass

    time_logs = {}
    measures = {}
    raw_predictions = {}
    for model_name in embedding_models:
        measures[model_name] = {}
        for feature in features:
            checkpoints_directory = f"{ckpt_dir}"
            if not os.path.exists(checkpoints_directory):
                Path.mkdir(Path(checkpoints_directory).parent, exist_ok=True)

            print("Current checkpoint directory: ", checkpoints_directory)
            config = _jsonnet.evaluate_file(
                CONFIG_PATH,
                ext_vars={
                    "model_name": model_name,
                    "feature": feature,
                    "checkpoints_directory": checkpoints_directory,
                },
            )
            model = EmbeddingsFactory(model_name=model_name, 
                                      checkpoints_directory= checkpoints_directory)
            config = json.loads(config)
            print(f"Training regressor for {model_name} - {feature}")

            train_checkpoint = checkpoints_directory + f"{model_name}_{feature}/"
            if not os.path.exists(train_checkpoint):
                Path.mkdir(Path(train_checkpoint), exist_ok=True)

            regressor, regressor_log = train_regressors(
                model=model,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                checkpoints_directory=train_checkpoint,
                config=config,
                use_cache=True,
            )
            for k in regressor_log.keys():
                time_logs[k] = time_logs.get(k, []) + [regressor_log[k]]

            test_dataset = pd.read_csv(path+'test.csv')
            # test_dataset = pd.read_csv(path+'dev.csv')
            dict_out = evaluate(
                regressor=regressor,
                model=model,
                config=config,
                eval_dataset=test_dataset,  # substitue with TEST once it's ready
                model_name=model_name,
                feature=feature,
                model_feature_name=model_name+'_'+feature
            )

            raw_predictions.update(dict_out['predictions'])
            print(f"MSE for {model_name} - {feature} is {dict_out['mse']}")

            for key in dict_out.keys():
                if key == 'predictions':
                    continue
                df_key = feature+'_'+key
                measures[model_name] |= {df_key: dict_out[key]}
                

    # Construct a dataframe that has for columns feature, model1, model2, ... modelN.
    # And every row is a feature+measure (MSE, MAE, Precision, Recall, F1)

    raw_predictions = pd.DataFrame(raw_predictions)
    raw_predictions.to_csv(ckpt_dir+'predictions_raw.csv', index=False)
    measures_df = pd.DataFrame(measures)
    measures_df.to_csv(ckpt_dir + "measures.csv")

    try:
        time_logs = pd.DataFrame(time_logs)
        time_logs.to_csv(ckpt_dir + "time_logs.csv")
    except:
        breakpoint()

    # filenames = test_dataset["filename"]
    # test_embeddings = get_embeddings(filenames)
