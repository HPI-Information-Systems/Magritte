"""Library of convenience functions/modules for Transformer encoders and AE models.
"""

import contextlib
import matplotlib.pyplot as plt
import logging
import math
from shutil import get_terminal_size

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from contextlib import suppress
import seaborn as sns

torch.set_printoptions(linewidth=get_terminal_size()[0])
logger = logging.getLogger(__name__)

def confusion_matrix_figure(predicted, target, vocab):
    """
    Plot a confusion matrix with an heatmap and returns the figure
    cm: confusion matrix where rows are target and columns are predicted
    """
    y_pred = predicted.astype(int).tolist()
    y_true = target.astype(int).tolist()
    labels = list(sorted(set(y_true)))
    labels += list(sorted([x for x in set(y_pred) if x not in labels]))
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    cm = cm[~np.all(cm == 0, axis=1)]

    cm_labels = vocab.lookup_tokens(labels)
    with contextlib.suppress(ValueError):
        cm_labels[cm_labels.index(" ")] = "(SPC)"
    with contextlib.suppress(ValueError):
        cm_labels[cm_labels.index("\t")] = "(TAB)"
    with contextlib.suppress(ValueError):
        cm_labels[cm_labels.index("[UNK]")] = "(None)"

    fig, ax = plt.subplots(figsize=(8, 8))
    # im = ax.matshow(matrix, cmap=plt.cm.Blues, alpha)
    sns.heatmap(cm, 
                cmap='binary', 
                linecolor='black',
                linewidths=.25,
                robust=True,
                xticklabels=cm_labels, 
                yticklabels=cm_labels[:len(cm[:,0])], 
                annot=True, 
                fmt='d', 
                cbar=False,
                square=True, 
                ax=ax
                )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")

    return fig


def f1_table(predicted_dict, target_dict):
    
    res = {}
    for k in predicted_dict.keys():
        p, r, f1, s = precision_recall_fscore_support(target_dict[k], predicted_dict[k], average="weighted", zero_division=0)
        res[k] = {"precision": p, "recall": r, "f1": f1, "support": s} 

    accuracy = 0
    n_samples = len(predicted_dict["delimiter"])
    for idx in range(n_samples):
        if predicted_dict["delimiter"][idx] == target_dict["delimiter"][idx] \
                and predicted_dict["quotechar"][idx] == target_dict["quotechar"][idx] \
                and predicted_dict["escapechar"][idx] == target_dict["escapechar"][idx]:
            accuracy += 1
    
    accuracy = accuracy / n_samples
    
    table = f"| Class | F1 | Precision | Recall | Support |\n"
    table += f"| --- | --- | --- | --- | --- |\n"
    for k in res.keys():
        table += f"| {k} | {res[k]['f1']:.4f} | {res[k]['precision']:.4f} | {res[k]['recall']:.4f} | {res[k]['support'] or ''} |\n"
    table += f"| Accuracy | {accuracy:.4f} | | | |\n"
    return table


def get_attn_pad_mask(seq_q, seq_k, PAD_INDEX=0):
    """This function generates a mask for the input tokens, containing 1 in the positions that are padded
    """
    len_q = seq_q.size()[-1]
    pad_attn_mask = seq_k.data.eq(PAD_INDEX)  # ... x 1 x len_k(=len_q), one is masking
    # return pad_attn_mask.unsqueeze(-2).repeat(*([1]*len(seq_q.size()[:-1])), len_q, 1)  # batch_size x len_q x len_k

def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))