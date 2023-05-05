"""
BERT code adapted from https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
VAE code adapted from https://github.com/shivakanthsujit/VAE-PyTorch
"""
import re
from typing import Dict, List

import torch
import torchvision
from torch import optim

from csv_embedder.magritte_base.model import MagritteBase
from .components import resnet18_decoder

class MagrittePretrainingVAE(MagritteBase):
    """
    Magritte Model with a head on top of the row embedders for the file encoding pretraining:
    a `variational autoencoder` head that trains on the reconstruction loss
    """

    def __init__(self,
                 beta: float = 4.0,
                 optimizer_lr = 1e-4,
                 no_grad: List[str] = [],
                 *args, **kwargs
                 ):
        super(MagrittePretrainingVAE, self).__init__(*args, **kwargs)

        self.decoder = resnet18_decoder(first_conv=True, 
                                        maxpool1=True, 
                                        latent_dim = self.encoding_dim, 
                                        input_height = 128)

        self.beta = beta
        self.no_grad = no_grad
        self.optimizer_lr = optimizer_lr

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    # @overrides
    def forward(self, input_tokens, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the self attention, the file embeddings
        """

        y = super(MagrittePretrainingVAE, self).forward(input_tokens, *args, **kwargs)
        row_embeddings = y["row_embeddings"]
        decoded_file = self.decoder(y["file_embedding"])
        decoded_file = decoded_file.permute(0, 2, 3, 1)
        row_embeddings = row_embeddings.permute(0, 2, 3, 1)

        return {"row_embeddings": row_embeddings,
                "decoded_file": decoded_file,
                }

    def configure_optimizers(self):

        for name, parameter in self.named_parameters():
            if any(re.search(regex, name) for regex in self.no_grad):
                parameter.requires_grad_(False)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.optimizer_lr)
        return optimizer

    def make_grid(self, output):
        real = output["row_embeddings"][:16, :, :, :3].permute(0, 3, 1, 2)
        fake = output["decoded_file"][:16, :, :, :3].permute(0, 3, 1, 2)
        grid_real = torchvision.utils.make_grid(real, normalize=False)
        grid_fake = torchvision.utils.make_grid(fake, normalize=False)
        return grid_fake, grid_real

    def step(self, batch):
        output = self.forward(batch)

        padding_mask = batch.data.eq(self.padding_index)
        padding_mask = padding_mask.unsqueeze(3).repeat((1, 1, 1, self.d_model))  # replicate along D dimension
        output["decoded_file"] = output["decoded_file"].masked_fill(padding_mask, 0)
        output["row_embeddings"] = output["row_embeddings"].masked_fill(padding_mask, 0)
        non_zero_elements = (~padding_mask).sum()

        MSE = torch.nn.MSELoss(reduction="none")(output["decoded_file"], output["row_embeddings"])
        mse_loss = (MSE * (~padding_mask).float()).sum() # gives \sigma_euclidean over unmasked elements
        mse_loss = mse_loss / non_zero_elements

        return output, mse_loss

    def training_step(self, batch, batch_idx):
        output, loss = self.step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 10 == 0:
            grid_fake, grid_real = self.make_grid(output)
            tensorboard = self.logger.experiment
            tensorboard.add_image("train_real", grid_real, self.current_epoch*10000+batch_idx)
            tensorboard.add_image("train_fake", grid_fake, self.current_epoch*10000+batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self.step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 10 == 0:
            grid_fake, grid_real = self.make_grid(output)
            tensorboard = self.logger.experiment
            tensorboard.add_image("val_real", grid_real, self.current_epoch*10000+batch_idx)
            tensorboard.add_image("val_fake", grid_fake, self.current_epoch*10000+batch_idx)

        return loss

    def on_train_epoch_end(self):
        path = self.save_path + "_current"
        torch.save(self.state_dict(), path)
