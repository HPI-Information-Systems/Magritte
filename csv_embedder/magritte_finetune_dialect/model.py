"""This file contains bare Magritte model with a classification head for dialect detection.
 The input to the classification head are the combination of file encodings with row CLS vectors.
 """

import os
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import F1Score

from torchtext.vocab import Vocab
from torch import optim

from csv_embedder.utils import confusion_matrix_figure, f1_table
from csv_embedder.magritte_base.model import MagritteBase

torch.set_float32_matmul_precision('medium')


class MagritteFinetuneDialectDetection(MagritteBase):

    def __init__(self,
                 label_vocab: Vocab,
                 n_classes: int,
                 weights: List[int],
                 optimizer_lr=1e-4,
                 *args, **kwargs
                 ):
        super(MagritteFinetuneDialectDetection, self).__init__(*args, **kwargs)
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.label_vocab = label_vocab
        self.tagger = nn.Linear(self.d_model * 2 + self.encoding_dim, self.n_classes)
        self.unk_idx = self.token_vocab["[UNK]"]

        pad_lbl = self.label_vocab(["[PAD]"])[0]
        weight = torch.tensor(weights, dtype=torch.float)
        print("Dialect vocabulary indices:", [(idx,s) for idx,s in enumerate(self.label_vocab.get_itos())])
        self.finetune_loss = nn.CrossEntropyLoss(ignore_index=pad_lbl, weight=weight)
        self.delimiter_loss = nn.CrossEntropyLoss()
        self.quotechar_loss = nn.CrossEntropyLoss()
        self.escapechar_loss = nn.CrossEntropyLoss()

        self.train_f1 = {}
        self.val_f1 = {}
        for k in ["delimiter", "quotechar", "escapechar"]:
            self.train_f1[f"f1_{k}"] = F1Score("multiclass", num_classes=len(self.token_vocab), average="micro").to(self.model_device)
            self.val_f1[f"f1_{k}"] = F1Score("multiclass", num_classes = len(self.token_vocab), average="micro").to(self.model_device)

        acc_keys = ["train_predicted_delimiter", "train_predicted_quotechar", "train_predicted_escapechar",
                    "train_target_delimiter", "train_target_quotechar", "train_target_escapechar",
                    "val_predicted_delimiter", "val_predicted_quotechar", "val_predicted_escapechar",
                    "val_target_delimiter", "val_target_quotechar", "val_target_escapechar"]

        self.accumulators = {key: torchmetrics.CatMetric() for key in acc_keys}
        self.optimizer_lr = optimizer_lr


    def extract_logits(self, input_tokens, tag_softmax, class_idx, padding_mask, vocab_size, batch_size):
        mask = (tag_softmax.argmax(dim=3).eq(class_idx))
        mask = torch.where(~padding_mask, mask, 0).bool()
        detected_tokens = torch.where(mask, input_tokens, self.unk_idx).view(batch_size, -1)

        # this function sets the logits of [UNK] to 0 if there is at least one detected character in the row
        masking = (detected_tokens == self.unk_idx).all(dim=1)
        logits = torch.stack([torch.bincount(row, minlength=vocab_size) for row in detected_tokens], dim=0)
        logits[:, self.unk_idx] *= masking
        logits = torch.softmax(logits.float(), dim=1)
        return logits


    def forward(self, input_tokens, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """

        :param input_tokens: the row pattern tokens as input to the model - as numeric indices of a vocabulary #todo annotate format?
        :param token_type_ids: the type of the token, used for encoding the same file objective function
        :return: dict containing the row embeddings, the self attention, the file embeddings
        """
        y = super(MagritteFinetuneDialectDetection, self).forward(input_tokens, *args, **kwargs)
        row_embeddings = y["row_embeddings"]
        file_embedding = y["file_embedding"]
        batch_size = row_embeddings.shape[0]

        # replicate file embedding to match the number of rows
        row_vecs = row_embeddings.permute(0, 2, 3, 1)
        row_cls = row_vecs[:, :, 0, :].unsqueeze(2).expand(batch_size, self.max_rows, self.max_len, -1)  # shape [batch_size, n_rows, d_model]
        file_vec = file_embedding.unsqueeze(1).unsqueeze(1).expand(batch_size, self.max_rows, self.max_len, -1)

        # concatenate the file_vec and row_cls embedding with the row vecs along last dimension
        dialect_embeddings = torch.cat((row_vecs, row_cls, file_vec), dim=3)
        tags_logits = self.tagger(dialect_embeddings)
        tag_softmax = nn.Softmax(dim=3)(tags_logits)

        del_class, quote_class, escape_class = self.label_vocab(["D", "Q", "E"])
        padding_mask = input_tokens.data.eq(self.padding_index)
        padding_mask = padding_mask +input_tokens.data.eq(self.cls_index)
        padding_mask = padding_mask + input_tokens.data.eq(self.sep_index)

        vocab_size = len(self.token_vocab)
        delimiter_logits = self.extract_logits(input_tokens, tag_softmax, del_class, padding_mask, vocab_size, batch_size)
        predicted_delimiter = torch.argmax(delimiter_logits, dim=1)

        quotechar_logits = self.extract_logits(input_tokens, tag_softmax, quote_class, padding_mask, vocab_size, batch_size)
        predicted_quotechar = torch.argmax(quotechar_logits, dim=1)
        
        escapechar_logits = self.extract_logits(input_tokens, tag_softmax, escape_class, padding_mask, vocab_size, batch_size)
        predicted_escapechar = torch.argmax(escapechar_logits, dim=1)

        return {"row_embeddings": row_embeddings,
                "file_embedding": file_embedding,
                "tags_logits": tags_logits,
                "predicted_tags": tag_softmax,
                "predicted_delimiter": predicted_delimiter,
                "predicted_quotechar": predicted_quotechar,
                "predicted_escapechar": predicted_escapechar,
                "delimiter_logits": delimiter_logits,
                "quotechar_logits": quotechar_logits,
                "escapechar_logits": escapechar_logits,
                }

    def training_step(self, batch, batch_idx):
        x, target = batch
        target_tags = target["target_tags"]
        output = self.forward(x)
        tags_logits = output["tags_logits"]
        delimiter_logits = output["delimiter_logits"]
        quotechar_logits = output["quotechar_logits"]
        escapechar_logits = output["escapechar_logits"]

        predicted = tags_logits.view([-1, self.n_classes])
        target_tags = target_tags.view(-1)
        tag_loss = self.finetune_loss(predicted, target_tags)
        del_loss = self.delimiter_loss(delimiter_logits, target["target_delimiter"])
        quote_loss = self.quotechar_loss(quotechar_logits, target["target_quotechar"])
        escape_loss = self.escapechar_loss(escapechar_logits, target["target_escapechar"])
        loss = tag_loss + del_loss + quote_loss + escape_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("tag_loss", tag_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("del_loss", del_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("quote_loss", quote_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("escape_loss", escape_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for k in ["delimiter", "quotechar", "escapechar"]:
            self.accumulators["train_predicted_" + k].update(output["predicted_" + k])
            self.accumulators["train_target_" + k].update(target["target_" + k])
            f1 = self.train_f1["f1_" + k](output["predicted_" + k], target["target_" + k])
            self.log("f1_"+k, f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def on_train_epoch_end(self):
        path = self.save_path + "_current.pth"
        print("Saving model in " + path)
        torch.save(self.state_dict(), path)

        [self.train_f1[k].reset() for k in self.train_f1.keys()]

        keys = ["train_predicted_delimiter", "train_predicted_quotechar", "train_predicted_escapechar",
                "train_target_delimiter", "train_target_quotechar", "train_target_escapechar"]
        y_del, y_quo, y_esc, t_del, t_quo, t_esc = [self.accumulators[k].compute().cpu().numpy() for k in keys]

        for k in keys:
            self.accumulators[k].reset()

        tensorboard = self.logger.experiment
        del_cm = confusion_matrix_figure(y_del, t_del, self.token_vocab)
        quo_cm = confusion_matrix_figure(y_quo, t_quo, self.token_vocab)
        esc_cm = confusion_matrix_figure(y_esc, t_esc, self.token_vocab)

        tensorboard.add_figure("train_del_CM", del_cm, self.current_epoch)
        tensorboard.add_figure("train_quo_CM", quo_cm, self.current_epoch)
        tensorboard.add_figure("train_esc_CM", esc_cm, self.current_epoch)

        table = f1_table({"delimiter":y_del,"quotechar":y_quo, "escapechar":y_esc},
                         {"delimiter":t_del,"quotechar":t_quo, "escapechar":t_esc})
        tensorboard.add_text("train_f1_table", table, self.current_epoch)


    def validation_step(self, batch, batch_idx):
        x, target = batch
        target_tags = target["target_tags"]
        output = self.forward(x)
        tags_logits = output["tags_logits"]
        delimiter_logits = output["delimiter_logits"]
        quotechar_logits = output["quotechar_logits"]
        escapechar_logits = output["escapechar_logits"]

        predicted = tags_logits.view([-1, self.n_classes])
        target_tags = target_tags.view(-1)
        tag_loss = self.finetune_loss(predicted, target_tags)
        del_loss = self.delimiter_loss(delimiter_logits, target["target_delimiter"])
        quote_loss = self.quotechar_loss(quotechar_logits, target["target_quotechar"])
        escape_loss = self.escapechar_loss(escapechar_logits, target["target_escapechar"])
        loss = tag_loss + del_loss + quote_loss + escape_loss

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_tag_loss", tag_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_del_loss", del_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_quote_loss", quote_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_escape_loss", escape_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for k in ["delimiter", "quotechar", "escapechar"]:
            self.accumulators["val_predicted_" + k].update(output["predicted_" + k])
            self.accumulators["val_target_" + k].update(target["target_" + k])
            f1 = self.val_f1["f1_" + k](output["predicted_" + k], target["target_" + k])
            self.log("val_f1_"+k, f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def on_validation_epoch_end(self):

        [self.val_f1[k].reset() for k in self.val_f1.keys()]
        keys = ["val_predicted_delimiter", "val_predicted_quotechar", "val_predicted_escapechar",
                "val_target_delimiter", "val_target_quotechar", "val_target_escapechar"]
        y_del, y_quo, y_esc, t_del, t_quo, t_esc = [self.accumulators[k].compute().cpu().numpy() for k in keys]
        for k in keys:
            self.accumulators[k].reset()

        del_cm = confusion_matrix_figure(y_del, t_del, self.token_vocab)
        quo_cm = confusion_matrix_figure(y_quo, t_quo, self.token_vocab)
        esc_cm = confusion_matrix_figure(y_esc, t_esc, self.token_vocab)

        tensorboard = self.logger.experiment
        tensorboard.add_figure("val_del_CM", del_cm, self.current_epoch)
        tensorboard.add_figure("val_quo_CM", quo_cm, self.current_epoch)
        tensorboard.add_figure("val_esc_CM", esc_cm, self.current_epoch)

        table = f1_table({"delimiter":y_del,"quotechar":y_quo, "escapechar":y_esc},
                         {"delimiter":t_del,"quotechar":t_quo, "escapechar":t_esc})
        tensorboard.add_text("val_f1_table", table, self.current_epoch)

    def predict_step(self, batch: Any, batch_idx: int=0, dataloader_idx: int = 0) -> Any:
        x, target = batch
        output = self.forward(x)
        return output, target

    def on_train_end(self):
        print("Saving model in " + self.save_path)
        torch.save(self.state_dict(), self.save_path)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
        return optimizer

    def load_weights(self, load_path=None):
        if load_path is not None and os.path.isfile(load_path):
            try:
                self.load_state_dict(torch.load(load_path))
            except RuntimeError:
                self.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
            print("Restored model from " + load_path)
        else:
            print("Magritte base model not found or load path not given.")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
