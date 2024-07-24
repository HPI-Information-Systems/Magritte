# MaGRiTTE
This repository contains the artifacts and source code for MaGRiTTE: a project to Machine Generate Representation of Tabular files with Transformer Encoders.

To reproduce the results of the paper, please follow the instructions in the following sections.

## Set up the environment
To setup the environment, we recommend using a virtual environment to avoid conflicts with other packages.
If using conda, run:

```conda env create -f environment.yml``` 

to create the environment, and then 

```conda activate magritte``` 

to activate it.
If using pip, run:

 `pip3 install -r requirements.txt`

## Use Case: Pollution Data
The input data to reproduce the use case presented in the paper is available in the `data/massbay` folder.
Two scripts can be used to integrate these files. 
 - A hand-crafted script can be launched with  ``python3 use_case_manual.py``. The script will generate the result file `results/massbay/manual_integrated.csv`.
 - The script that uses MaGRiTTE can be launched with ``python3 use_case_magritte.py``. The script will generate the result file `results/massbay/magritte_integrated.csv`.

Running the MaGRiTTE version of the script requires downloading the corresponding weights for the model. 
The weights can be downloaded at [url](https://mega.nz/file/wJMhDbIa#Zmr23xd67xktcZtpvuu781Om2uwb1PWtQina6a_zwKg) See also section below.

## Datasets
The folder `data` contains the datasets used for the experiments. The datasets are organized in subfolders, each containing the data for a specific task. 
Due to the space policy of GitHub, we publish some of our datasets in compressed folders in ``tar.gz`` formats, or on external servers.
To simplify download and extraction, we provide a script to automatically downloads and extracts the data (``download_data.sh``)
Run the ``download_data.sh`` script in the root repository folder to download and extract the data (requires a *nix system with ``megatools`` installed)

The datasets are organized as follows:
- `/gittables`: contains the data from the gittables dataset, organized for the pretraining task. To download and extract the dataset, run `download.sh` in the folder0.
- `dialect_detection`: contains the data for the finetuning task for dialect detection. To extract the files automatically, run `extract.sh` in the folder.
- `row_classification`: contains the data for the finetuning task for row classification. Every dataset is contained in a separate ``tar.gz`` file, and their annotations are contained in the jsonl files `train_dev_annotations.jsonl` and `test_annotations.jsonl`.
- `estimate`: contains the data for the finetuning task for estimate.
- `columntype`: contains the data for the finetuning task for column type detection classification. There are three versions of the dataset, one for each scenario: (1) the `unprepared` folder contains raw files, (2) the `autoclean` folder contains the files after automated cleaning with MaGRiTTE, and (3) the `clean` folder contains the ground truth cleaned files. The annotations for the column types of each dataset are contained in `csv` files in each folder.
 
## Model Weights
The folder `weights` contains the weights for the model used for the experiments. 
Due to the space policy of GitHub, we publish the weights on external servers.
The main weights can be found at [https://mega.nz/file/wJMhDbIa#Zmr23xd67xktcZtpvuu781Om2uwb1PWtQina6a_zwKg](https://mega.nz/file/wJMhDbIa#Zmr23xd67xktcZtpvuu781Om2uwb1PWtQina6a_zwKg).
To simplify download and extraction, we provide a script to automatically downloads and extracts the data (``download_weights.sh``)
Run the ``download_weights.sh`` script in the root repository folder to download and extract the weights (requires a *nix system with ``megatools`` installed).
Alternatively, the weights can be manually downloaded from the links provided in the `links.txt` file.

The `data` folder contains the data used for the experiments, arranged in several subfolders. 


## Training the MaGRiTTE model

 - Once the environment is set up and the data has been downloaded, the three folders `configs`, `embedder`, and `training` can be used to pretrain/finetune MaGRiTTE on several tasks.
 - `configs`: contains the configuration files in .jsonnet format used for to train the models and run the experiments
 - `embedder`: contains the source code of the MaGRiTTE model, organized in subfolders depending on the pretraining/finetuning tasks.
 - `training`: contains the scripts to train the models. 
 - `data`: contains the datasets with the ground truths for the finetuning tasks.
 - `experiments`: contains the scripts to run the experiments for testing purposes after finetuning.
 - `weights`: contains the weights of the models used for the paper. 
 
 For example, to finetune the model to the dialect detection task, it is sufficient to run:
 
  ``python3 training/train_dialect.py`` 
  
  which reads the configuration file stored in ``configs/dialect.jsonnet``.

Each training will saves intermediate artifacts in a corresponding ``tensorboard`` folder under ``results\{pretrain,dialect,rowclass,estimate}``. To visualize the state of training, you can run e.g. 

``tensorboard --logdir results\pretrain\tensorboard`` 

and open the browser at ``localhost:6006``.
The model weights are saved in the folder ``weights``. If you would like to skip the training phase, we provide the weights of the models used for the paper. Refer to the section ``Model Weights`` for instructions on how to download the weights.

## Running the experiments
 The `experiments` folder contains the scripts to run the experiments for testing purposes after finetuning. The scripts are organized in subfolders depending on the finetuning tasks.
 Each scripts loads the corresponding trained model from the ``weights`` folder and runs the experiments on the dev/test set. 
To run the experiments, run e.g. 

``python3 experiments/dialect/magritte.py``

 The results for each task are saved in the folder ``results``, under a corresponding subfolder. 
 
 The folder ``plots`` can be used after the experiments to generate the plots for the paper. The plots are generated using Jupyter notebooks, one for each of the tasks, which read the results from the ``results`` folder and save the corresponding image in ``.png`` format.