import re
from typing import Dict, Any, List
from matplotlib import pyplot as plt

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torchmetrics
from torch import optim

from csv_embedder.magritte_base.model import MagritteBase

torch.set_float32_matmul_precision('medium')



def scatter_figure(predicted, target):
    y_pred = predicted.tolist()
    y_true = target.tolist()

    labelsize = 20
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(zorder=-10)
    ax.set_axisbelow(True)
    bins = [np.arange(-.05,1.15,.1), np.arange(-.05,1.15,.1)]
    ax.hist2d(y_true,y_pred, cmap = "OrRd", bins = bins, zorder = -13)
    ax.scatter(y_true,y_pred, zorder=1, color =  "#D81B60")
    ax.set_xlabel("Target", size=labelsize)
    ax.set_ylabel("Estimate", size=labelsize)
    ticks = list(np.arange(0,1.1,0.1))
    ticklabels = [f"{tick:.1f}" for tick in ticks]
    ax.set_xticks(ticks=ticks, labels=ticklabels, size=labelsize-2);
    ax.set_yticks(ticks=ticks, labels=ticklabels, size=labelsize-2);
    return fig


class MagritteFinetuneEstimation(MagritteBase):

    def __init__(self, optimizer_lr=1e-4, no_grad: List[str] = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.row_embedder = nn.Linear(self.d_model, 1) 
        self.estimator = nn.Linear(self.encoding_dim*4, 1)
        self.no_grad = no_grad

        self.optimizer_lr = optimizer_lr
        self.regression_loss = nn.MSELoss()
        self.float()

        acc_keys = ["train_predicted", "train_target", "val_predicted", "val_target"]
        self.accumulators = {key: torchmetrics.CatMetric() for key in acc_keys}
        print("Encoding dim is " + str(self.encoding_dim))

    def forward(
        self, input_tokens_a, input_tokens_b, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        :param input_tokens_a: the row pattern tokens from the first file
        :param input_tokens_b: the row pattern tokens frin the second file
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the self attention, the file embeddings
        """

        y_a = super().forward(input_tokens_a, *args, **kwargs)
        file_embedding_a = y_a["file_embedding"]

        y_b = super().forward(input_tokens_b, *args, **kwargs)
        file_embedding_b = y_b["file_embedding"]

        cls_a = y_a["row_embeddings"].permute(0,2,3,1)[:,:,0,:]
        cls_b = y_b["row_embeddings"].permute(0,2,3,1)[:,:,0,:]
        
        synth_row_embedding_a = self.row_embedder(cls_a).squeeze(2)
        synth_row_embedding_b = self.row_embedder(cls_b).squeeze(2)

        enc = torch.cat([file_embedding_a, 
                        synth_row_embedding_a,
                        file_embedding_b,
                        synth_row_embedding_b], dim=1)
        enc = self.estimator(enc)
        estimate = nn.Sigmoid()(enc).squeeze(1)

        return {
            "file_embedding_a": file_embedding_a,
            "file_embedding_b": file_embedding_b,
            "estimate": estimate,
        }

    def training_step(self, batch, batch_idx):
        file_a, file_b, target = batch
        output = self.forward(file_a, file_b)
        estimate = output["estimate"]

        loss = self.regression_loss(estimate.float(), target.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if estimate.isnan().any():
            estimate = torch.nan_to_num(estimate, nan=1, posinf=1, neginf=0)
        self.accumulators["train_predicted"].update(estimate.detach())
        self.accumulators["train_target"].update(target.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        file_a, file_b, target = batch
        output = self.forward(file_a, file_b)
        estimate = output["estimate"]
        if estimate.isnan().any():
            estimate = torch.nan_to_num(estimate, nan=1, posinf=1, neginf=0)

        loss = self.regression_loss(estimate.float(), target.float())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.accumulators["val_predicted"].update(estimate.detach())
        self.accumulators["val_target"].update(target.detach())
        return loss

    def predict_step(self, batch: Any, batch_idx: int=0, dataloader_idx: int = 0) -> Any:
        file_a, file_b, target = batch
        output = self.forward(file_a, file_b)
        return output["estimate"]
    
    def on_train_epoch_end(self) -> None:
        path = self.save_path + "_current"
        print("Saving model in " + path)
        torch.save(self.state_dict(), path)

        tensorboard = self.logger.experiment
        y_pred = self.accumulators["train_predicted"].compute().cpu().numpy()
        y_true = self.accumulators["train_target"].compute().cpu().numpy()
        scatter = scatter_figure(y_pred, y_true)
        tensorboard.add_figure("train_scatter", scatter, self.current_epoch)

        [self.accumulators[k].reset() for k in self.accumulators.keys() if "train" in k]

    def on_validation_epoch_end(self) -> None:

        y_pred = self.accumulators["val_predicted"].compute().cpu().numpy()
        y_true = self.accumulators["val_target"].compute().cpu().numpy()

        scatter = scatter_figure(y_pred, y_true)
        tensorboard = self.logger.experiment
        tensorboard.add_figure("val_scatter", scatter, self.current_epoch)

        [self.accumulators[k].reset() for k in self.accumulators.keys() if "val" in k]


    def configure_optimizers(self):
        for name, parameter in self.named_parameters():
            if any(re.search(regex, name) for regex in self.no_grad):
                parameter.requires_grad_(False)

        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
        return optimizer
