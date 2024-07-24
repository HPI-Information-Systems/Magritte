import os
import lightning.pytorch as pl
import torchmetrics
import torch
from tqdm import tqdm
class EmbeddingsModel:
    """ A class interface for all different embedding models.
    In the __init__ method the model should be loaded and stored in the self.model attribute."""
    
    base_index = 0
    subset = None
    embtype = "file"

    def get_embeddings(self, model_name, 
                       filenames, 
                       dataset_name, 
                       filedir="/data/survey/csv/"):
        embedding_path = os.path.join(
        self.checkpoints_directory, f"{model_name}_{dataset_name}_embeddings.pth"
        )
        if os.path.exists(embedding_path):
            embeddings = torch.load(embedding_path)
            return embeddings
        else:
            embeddings = []
            for f in tqdm(filenames[self.base_index:self.subset]):
                curr_filepath = os.path.join(filedir, f)
                if self.embtype == "file":
                    embedding = self.model.embed(curr_filepath)["file_embedding"]
                elif self.embtype == "row":
                    embedding = self.model.embed(curr_filepath)["row_embeddings"] # shape [batch_size, d_model, n_rows, row_len]
                    embedding = embedding[:,0,:,:] # select CLS token
                    embedding = torch.mean(embedding, dim=1)
                    assert embedding.shape[1] == self.model.d_model
                else:
                    raise ValueError("Invalid embedding type")

                try:
                    if embedding.shape[0] == 1:
                        embedding = embedding.squeeze()
                except:
                    pass
                embeddings.append(embedding)
            embeddings = torch.stack(embeddings)
            torch.save(embeddings, embedding_path)

            return embeddings


class AblationModel(pl.LightningModule):
    """
        Linear layer, the only relevant parameter is the embedding size which needs to be the sme as the input table embeddings.
    """
    def __init__(self, 
                 embedding_size, 
                 optimizer_lr, 
                 feature="n_rows",
                 pos_weights = None):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer_lr = optimizer_lr
        self.feature = feature
        if feature in ("n_rows", "n_cols"):
            self.loss = torch.nn.functional.mse_loss
            self.f1 = torchmetrics.MeanSquaredError().to(self.device)
        else:
            self.loss = torch.nn.BCELoss()
            self.f1 = torchmetrics.F1Score(task="binary").to(self.device)
        self.pos_weights = pos_weights

    def forward(self, inputs):
        x = self.linear(inputs)
        # x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs).view(-1)
        reshape_target = target.reshape(output.shape)
        with torch.autocast(device_type='cuda', enabled=False):
            if self.pos_weights is not None:
                neg_weights = 1 - self.pos_weights
                weights = torch.tensor(self.pos_weights*target)
                weights = torch.where(target == 0, neg_weights, weights)
                loss = torch.nn.BCEWithLogitsLoss(weight=weights)(output.view(-1), reshape_target.view(-1))
            else:
                loss = self.loss(output, reshape_target)
        self.log("train_loss", loss, prog_bar=True)
        binary_output = torch.where(output > 0.5, 1, 0).to(self.device)
        self.log("train_f1", self.f1(binary_output, reshape_target), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        reshape_target = target.reshape(output.shape)
        with torch.autocast(device_type='cuda', enabled=False):
            if self.pos_weights is not None:
                neg_weights = 1 - self.pos_weights
                weights = torch.tensor(self.pos_weights*target)
                weights = torch.where(target == 0, neg_weights, weights)
                loss = torch.nn.BCEWithLogitsLoss(weight=weights)(output.view(-1), reshape_target.view(-1))
            else:
                loss = self.loss(output, reshape_target)

        self.log("val_loss", loss, prog_bar=True)
        binary_output = torch.where(output > 0.5, 1, 0).to(self.device)
        self.log("val_f1", self.f1(binary_output, reshape_target), prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs, target = batch
        output = self.forward(inputs)
        return output, target

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = torch.nn.functional.relu(self.linear(inputs))
        loss = self.loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)
