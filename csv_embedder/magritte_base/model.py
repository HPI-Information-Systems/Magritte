import os
from typing import Dict, Any
import torch
import torch.nn as nn

from torchtext.vocab import Vocab
from torch import optim
import lightning.pytorch as pl

from .components import resnet18_encoder
torch.set_float32_matmul_precision('medium')


class MagritteBase(pl.LightningModule):

    def __init__(self,
                 token_vocab: Vocab,
                 max_rows: int,
                 max_len: int,
                 n_layers: int,
                 n_heads: int,
                 d_model: int, d_k: int, d_v: int, d_ff: int,
                 n_segments: int,
                 encoding_dim: int,
                 save_path: str = "",
                 *args, **kwargs
                 ):
        super(MagritteBase, self).__init__()
        vocab_size_tokens = len(token_vocab)

        self.token_vocab = token_vocab
        self.max_len = max_len
        self.max_rows = max_rows
        self.encoding_dim = encoding_dim

        self.padding_index,\
        self.cls_index,\
        self.sep_index,\
        self.mask_index = token_vocab(["[PAD]", "[CLS]", "[SEP]", "[MASK]"])

        self.token_embedding = nn.Embedding(vocab_size_tokens, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.encoding_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                       activation='gelu',
                                       batch_first=True), n_layers)

        # Input: batch_size x channels_img x n_rows x max_tokens
        # Input: B_s x Dx64x64, output: 4x32x32
        self.d_model = d_model
        self.encoding_dim = encoding_dim
        self.encoder = resnet18_encoder(first_conv=True, maxpool1=True)
        self.hidden_dim = 512
        self.fc_latent = nn.Linear(self.hidden_dim, self.encoding_dim)

        self.n_heads = n_heads
        self.save_path = save_path
        print("When initializing, save path is ", save_path)

    def forward(self, input_tokens, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the file embeddings
        : shape of row_embeddings: (batch_size, n_rows, max_len, d_model)
        """

        batch_size = input_tokens.size(0)
        token_type_ids = torch.zeros_like(input_tokens)
        #if there is not at least one row without padding, we include one 
        if not (input_tokens.view(batch_size*self.max_rows,-1)[:,-1]).any(): 
            input_tokens[-1,-1,-1] = self.sep_index

        ret = torch.arange(self.max_len, dtype=torch.long).to(input_tokens.device)
        ret = ret.unsqueeze(0).expand_as(input_tokens)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.token_embedding(input_tokens) + self.position_embedding(ret) + self.segment_embedding(token_type_ids)
        row_embeddings = self.norm(embedding)

        batched_row_embeddings = row_embeddings.view(batch_size * self.max_rows, self.max_len, -1)
        pad_attn_mask = input_tokens.view(batch_size * self.max_rows, -1).data.eq(self.padding_index)
        batched_row_embeddings = self.encoding_layers(batched_row_embeddings, src_key_padding_mask=pad_attn_mask)
        row_embeddings = batched_row_embeddings.view(batch_size, self.max_rows, self.max_len, -1)

        row_embeddings = torch.permute(row_embeddings, (0, 3, 1, 2))
        x = self.encoder(row_embeddings).view(batch_size, -1)
        z = self.fc_latent(x)

        return {"row_embeddings": row_embeddings,  # shape [batch_size, d_model, n_rows, row_len]
                "file_embedding": z,}

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

    def on_train_end(self):
        print("Saving model in " + self.save_path)
        torch.save(self.state_dict(), self.save_path)

    def configure_optimizers(self):
        raise NotImplementedError

    def load_weights(self, load_path=None):
        if load_path is None:
            load_path = self.save_path
        if os.path.isfile(load_path):
            print("Restoring model from " + load_path)
            try:
                self.load_state_dict(torch.load(load_path))
            except RuntimeError:
                self.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
        else:
            print("Magritte base model not found")

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Same as allennlp load_state_dict except that if the weight dictionary comes from a
        super model from Magritte we extract only the base weights WARNING: strong coupling here.
        """
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
        print("Missing keys: ", len(missing_keys))
        print("Unexpected keys: ", len(unexpected_keys))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
