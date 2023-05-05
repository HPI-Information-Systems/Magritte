"""
BERT code adapted from https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
"""
import logging
import pdb
from typing import Dict, Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, F1Score
from torch import optim
from torch.optim.lr_scheduler import LinearLR
from torchtext.vocab import Vocab

from csv_embedder.utils import gelu
import os

class MagrittePretrainingRowPair(pl.LightningModule):

    def __init__(self,
                 vocab: Vocab,
                 max_len: int,
                 n_layers: int,
                 n_heads: int,
                 d_model: int, d_k: int, d_v: int, d_ff: int,
                 n_segments: int,
                 save_path: str = "",
                 load_path: str = None,
                 optimizer_lr: float = 1e-4,
                 *args, **kwargs
                 ):

        super(MagrittePretrainingRowPair, self).__init__()
        vocab_size = len(vocab)
        self.vocab = vocab
        self.save_path = save_path
        self.max_len = max_len
        self.padding_index = vocab(["[PAD]"])[0]
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.encoding_layers = nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                                           activation='gelu',
                                                           batch_first=True), n_layers)

        # Classification head
        self.pretrain_fc = nn.Linear(d_model, d_model)
        self.pretrain_activ1 = nn.Tanh()
        self.pretrain_classifier = nn.Linear(d_model,1)

        # Masked language model head
        self.pretrain_linear = nn.Linear(d_model, d_model)
        self.pretrain_activ2 = gelu
        self.pretrain_norm = nn.LayerNorm(d_model)

        # decoder is shared with embedding layer
        embed_weight = self.token_embedding.weight
        n_vocab, n_dim = embed_weight.size()
        self.pretrain_decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.pretrain_decoder.weight = embed_weight
        self.pretrain_decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        self.optimizer_lr = optimizer_lr
        self.masked_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.clsf_loss = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()

        self.train_lm_accuracy = Accuracy("multiclass", num_classes=vocab_size) # num_classes=self.vocab.get_vocab_size())
        self.train_clsf_f1 = F1Score("binary")

        self.val_lm_accuracy = Accuracy("multiclass", num_classes=vocab_size) # num_classes=self.vocab.get_vocab_size())
        self.val_clsf_f1 = F1Score("binary")

        if load_path is not None:
            self.load_weights(load_path)

    def forward(self, input_tokens, token_type_ids, masked_positions=None, **kwargs) -> Dict[str, torch.Tensor]:
        if torch.max(input_tokens)>=len(self.vocab):
            print("Something will be broken with vocabulary")
            pdb.set_trace()

        seq_len = input_tokens.size(1)
        ret = torch.arange(seq_len, dtype=torch.long).to(input_tokens.device)
        ret = ret.unsqueeze(0).expand_as(input_tokens)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.token_embedding(input_tokens) + self.position_embedding(ret) + self.segment_embedding(token_type_ids)
        row_embeddings = self.norm(embedding) # (batch_size, seq_len, d_model)
        pad_attn_mask = input_tokens.data.eq(self.padding_index)
        row_embeddings = self.encoding_layers(row_embeddings, src_key_padding_mask=pad_attn_mask)

        # Classification logits
        encoded_cls = row_embeddings[:,0,:]  # [batch_size, d_model]
        x = self.pretrain_fc(encoded_cls) #[batch_size, d_model]
        x = self.pretrain_activ1(x)  # [batch_size, d_model]
        x = self.pretrain_classifier(x)  # [batch_size, 1]
        logits_clsf = x.squeeze()  # [batch_size]

        # embeddings : [batch_size, len, d_model],
        # attn : [batch_size, n_heads, d_mode, d_model]
        padding_mask = masked_positions.data.eq(-100)
        masked_positions = masked_positions.masked_fill(padding_mask,0).unsqueeze(2)
        masked_positions = masked_positions.expand(-1, -1, self.d_model)  # [batch_size, max_pred, d_model]

        masked_tokens_encoding = torch.gather(row_embeddings, 1, masked_positions)  # [batch_size, max_pred, d_model]
        masked_tokens_encoding = self.pretrain_activ2(self.pretrain_linear(masked_tokens_encoding))
        masked_tokens_encoding = self.pretrain_norm(masked_tokens_encoding)
        logits_lm = self.pretrain_decoder(masked_tokens_encoding) + self.pretrain_decoder_bias  # [batch_size, max_pred, n_vocab]

        return {"input_ids": input_tokens, "masked_positions": masked_positions, "padding_mask": padding_mask,
                  "embeddings": row_embeddings, "logits_lm": logits_lm, "logits_clsf": logits_clsf}


    def calculate_loss(self, batch, batch_idx):
        x, target = batch
        output = self.forward(**x)

        masked_tokens = target["masked_tokens"]
        same_file = target["same_file"]

        logits_lm = output["logits_lm"]
        logits_clsf = output["logits_clsf"]

        padding_mask = x["masked_positions"].data.eq(-100)
        masked_tokens = masked_tokens.masked_fill(padding_mask.squeeze(), -100)
        loss_lm = self.masked_loss(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = self.clsf_loss(logits_clsf, same_file.float())  # for sentence classification
        return output, loss_lm, loss_clsf


    def training_step(self, batch, batch_idx):
        x, target = batch
        output, loss_lm, loss_clsf = self.calculate_loss(batch, batch_idx)
        loss = loss_lm + loss_clsf
        # reporting
        self.log("loss_lm", loss_lm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss_clsf", loss_clsf, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        logits_lm = output["logits_lm"]
        logits_clsf = output["logits_clsf"]
        acc = self.train_lm_accuracy(logits_lm.view(-1,len(self.vocab)), target["masked_tokens"].view(-1))
        f1 = self.train_clsf_f1(logits_clsf, target["same_file"])
        self.log("train_lm_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_clsf_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.train_lm_accuracy.reset()
        self.train_clsf_f1.reset()

        path = self.save_path+"_epoch"+str(self.current_epoch)
        print("Saving model in " + path)
        torch.save(self.state_dict(), path)

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output, loss_lm, loss_clsf = self.calculate_loss(batch, batch_idx)
        loss = loss_lm + loss_clsf
        # reporting
        self.log("val_loss_lm", loss_lm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_clsf", loss_clsf, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        logits_lm = output["logits_lm"]
        logits_clsf = output["logits_clsf"]
        acc = self.val_lm_accuracy(logits_lm.view(-1,len(self.vocab)), target["masked_tokens"].view(-1))
        f1 = self.val_clsf_f1(logits_clsf, target["same_file"])
        self.log("val_lm_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_clsf_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        self.val_lm_accuracy.reset()
        self.val_clsf_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
        return optimizer

    def load_weights(self, load_path = None):
        if load_path is None:
            load_path = self.save_path
        if os.path.isfile(load_path):
            print("Restoring model from " + load_path)
            wdict = torch.load(load_path)
            if wdict["position_embedding.weight"].shape != self.position_embedding.weight.shape:
                print("Position embedding size is different, not loading")
                wdict["position_embedding.weight"] = self.position_embedding.weight
            self.load_state_dict(wdict)
        else:
            print("Magritte base model not found in "+load_path)

    def on_train_end(self):
        print("Saving model in " + self.save_path)
        torch.save(self.state_dict(), self.save_path)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, target = batch
        output_dict = self.forward(**x)

        padding_mask = output_dict["padding_mask"].squeeze(2)

        predicted_mask_ids = torch.argmax(output_dict["logits_lm"], 2)  # [n_instances, n_masked_predictions]
        decoded_masks = []
        for instance_idx in predicted_mask_ids:
            decoded_masks.append(
                [self.vocab.get_token_from_index(predicted_idx.item())
                 for predicted_idx in instance_idx])

        tokens = []

        for instance_tokens in output_dict["input_ids"]:
            tokens.append(
                [self.vocab.get_token_from_index(token_id.item())
                 for token_id in instance_tokens])

        for instance_idx,instance_pos in enumerate(output_dict["masked_positions"]):
            for jdx,mask_pos in enumerate(instance_pos):
                tokens[instance_idx][mask_pos[0].item()] = decoded_masks[instance_idx][jdx]

        output_dict["predicted_words"] = decoded_masks
        output_dict["out_tokens"] = tokens

        return output_dict