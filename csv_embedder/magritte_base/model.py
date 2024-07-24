import os
from typing import Dict, Any
import torch
import torch.nn as nn

from torchtext.vocab import Vocab
from torchtext.vocab import vocab as build_vocab
from torch import optim
import lightning.pytorch as pl

from csv_embedder.pattern_tokenizer import PatternTokenizer
from .components import resnet18_encoder
torch.set_float32_matmul_precision('medium')


class MagritteBase(pl.LightningModule):

    def __init__(
        self,
        max_rows: int,
        max_len: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        n_segments: int,
        encoding_dim: int,
        save_path: str = "",
        nocnn: bool = False,
        reduce_features: bool = False,
        tokenizer = None,
        device="cuda",
        token_vocab: Vocab = None,
        vocab_path=None,
        *args,
        **kwargs
    ):
        super(MagritteBase, self).__init__()

        if token_vocab is None:
            assert vocab_path is not None
            tokens = open(vocab_path).read().splitlines()
            tokens[tokens.index("")] = "\n"
            ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
            self.token_vocab = build_vocab(ordered_tokens)
            self.token_vocab.set_default_index(self.token_vocab["[UNK]"])          
        else:
            self.token_vocab = token_vocab

        vocab_size_tokens = len(self.token_vocab)

        self.max_len = max_len
        self.max_rows = max_rows
        self.encoding_dim = encoding_dim

        self.padding_index,\
        self.cls_index,\
        self.sep_index,\
        self.mask_index = self.token_vocab(["[PAD]", "[CLS]", "[SEP]", "[MASK]"])

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
        
        self.nocnn = nocnn
        self.reduce_features = reduce_features
        if not nocnn:
            self.feature_embed = nn.Linear(self.d_model, 3)
            if reduce_features:
                self.encoder = resnet18_encoder(
                    input_depth=32, first_conv=True, maxpool1=True
                )
            else:
                self.encoder = resnet18_encoder(
                    input_depth=d_model, first_conv=True, maxpool1=True
                )
            self.hidden_dim = 512
            self.fc_latent = nn.Linear(self.hidden_dim, self.encoding_dim)

        self.n_heads = n_heads
        self.save_path = save_path
        # print("When initializing, save path is ", save_path)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = PatternTokenizer(
                token_vocab=self.token_vocab,
                max_len=self.max_len,
                max_rows=self.max_rows,
                padding_index=self.padding_index,
                cls_index=self.cls_index,
                sep_index=self.sep_index,
                mask_index=self.mask_index,
            )

        if not torch.cuda.is_available():
            self.to('cpu')
        else:
            self.to(device)


    def forward(self, input_tokens, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the file embeddings
        : shape of row_embeddings: (batch_size, n_rows, max_len, d_model)
        """

        batch_size = input_tokens.size(0)
        token_type_ids = torch.zeros_like(input_tokens)
        # if there is not at least one row without padding, we include one
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

        if self.reduce_features:
            embs = row_embeddings.view(batch_size, 128 * 128, self.d_model)
            encoder_embeddings = torch.nn.MaxPool1d(kernel_size=24)(embs)
            encoder_embeddings = encoder_embeddings.view(batch_size, 128, 128, 32)
        else:
            encoder_embeddings = row_embeddings

        encoder_embeddings = torch.permute(encoder_embeddings, (0, 3, 1, 2))
        output = {
            "row_embeddings": row_embeddings,  # shape [batch_size, d_model, n_rows, row_len]
            "pad_mask": pad_attn_mask,
            "encoder_embeddings": encoder_embeddings,
        }

        if not self.nocnn:
            x = self.encoder(encoder_embeddings).view(batch_size, -1)
            z = self.fc_latent(x)
            output["file_embedding"] = z

        return output
    def embed(self, filepath: str) -> Dict[str, torch.Tensor]:
        """
        Embeds a file using the model, first tokenizing it and then passing it through the model
        :param file: the file to embed
        :return: the embeddings of the file
        """
        with torch.no_grad():
            tokens = self.tokenizer.tokenize_file(filepath).type(torch.long).to(self.device)
            input_tokens = tokens.unsqueeze(0).to(self.device)
            return self.forward(input_tokens)

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
        # print("Missing keys: ", len(missing_keys))
        # print("Unexpected keys: ", len(unexpected_keys))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
