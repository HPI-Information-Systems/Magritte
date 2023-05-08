# Magritte
This repository contains the artifacts and source code for MaGRiTTE: a project to Machine Generate Representation of Tabular files with Transformer Encoders.

To reproduce the results of the paper, please follow the instructions in the following sections.

0. Setup environment and download datasets
To setup the environment, we recommend using a virtual environment to avoid conflicts with other packages.
If using conda, run `conda env create -f environment.yml` to create the environment, and then `conda activate magritte` to activate it.
If using pip, run `pip3 install -r requirements.txt`

The `data` folder contains the data used for the experiments, arranged in several subfolders. 
Due to the space policy of GitHub, we publish our datasets in compressed folders in ``tar.gz`` formats, or alternatively, with scripts that download and extract the relative files from the HPI servers. 
- `/gittables`: contains the data from the gittables dataset, organized for the pretraining task. To download and extract the dataset, run `download.sh` in the folder.
- `/dialect_detection`: contains the data for the finetuning task for dialect detection. To extract the files, run `extract.sh` in the folder.
- `estimate`: contains the data for the finetuning task for estimate. Due to licensing issues on some of the original csv/json files, we only share our annotations in a set of csv files used for the training, development and testing.
- `row_classification`: contains the data for the finetuning task for row classification. Every dataset is contained in a separate ``tar.gz`` file, and their annotations are contained in the jsonl files `train_dev_annotations.jsonl` and `test_annotations.jsonl`.
 

1. Training the MaGRiTTE model

 - Once the environment is set up and the data has been downloaded, the three folders `configs`, `csv_embedder`, and `training` can be used to pretrain/finetune MaGRiTTE on several tasks.
 - `configs`: contains the configuration files in .jsonnet format used for to train the models and run the experiments
 - `csv_embedder`: contains the source code of the MaGRiTTE model, organized in subfolders depending on the pretraining/finetuning tasks.
 - `training`: contains the scripts to train the models. 
 
 For example, to finetune the model to the dialect detection task, it is sufficient to launch ``python3 training/train_dialect.py`` which reads the configuration file stored in ``configs/dialect.jsonnet``.

Each training will saves intermediate artifacts in a corresponding ``tensorboard`` folder under ``results\{pretrain,dialect,rowclass,estimate}``. To visualize the state of training, you can run e.g. ``tensorboard --logdir results\pretrain\tensorboard`` and open the browser at ``localhost:6006``.
The model weights are saved in the folder ``weights``. If you would like to skip the training phase, we provide the weights of the models used for the paper. To use them, run the ``download.sh`` script in the ``weights`` folder that automatically downloads the weights from the HPI server.

2. Running the experiments
 The `experiments` folder contains the scripts to run the experiments for testing purposes after finetuning. The scripts are organized in subfolders depending on the finetuning tasks.
 Each scripts loads the corresponding trained model from the ``weights`` folder and runs the experiments on the dev/test set. 
To run the experiments, run e.g. ``python3 experiments/dialect/magritte.py``.
 The results for each task are saved in the folder ``results``, under a corresponding subfolder. 
 
 The folder ``plots`` can be used after the experiments to generate the plots for the paper. The plots are generated using Jupyter notebooks, one for each of the tasks, which read the results from the ``results`` folder and save the corresponding image in ``.png`` format.