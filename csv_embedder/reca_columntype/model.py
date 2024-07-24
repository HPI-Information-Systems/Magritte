import pdb
from transformers import BertModel, BertTokenizer
import os
import torch
from torchtext.vocab import vocab as build_vocab
import lightning.pytorch as pl
from torchmetrics.classification import F1Score
from torchmetrics import CatMetric
from sklearn.metrics import f1_score as sk_f1_score


class KREL(pl.LightningModule):
    def __init__(self, 
                 n_classes=116, 
                 d_model=768, 
                 optimizer_lr=1e-5, 
                 max_len=512,
                 device="cuda",
                 label_vocab=None,
                 label_path = None,
                 save_path=""):
        super(KREL, self).__init__()
        self.model_name = "KREL"
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.fcc_tar = torch.nn.Linear(d_model, n_classes)
        self.fcc_rel = torch.nn.Linear(d_model, n_classes)
        self.fcc_sub = torch.nn.Linear(d_model, n_classes)
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(1)) for i in range(3)]
        )
        self.n_classes = n_classes

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer_lr = optimizer_lr
        self.weighted_f1 = F1Score(
            task="multiclass", average="weighted", num_classes=n_classes
        )
        self.macro_f1 = F1Score(
            task="multiclass", average="macro", num_classes=n_classes
        )

        self.accumulators = {
            "train_predicted": CatMetric(),
            "train_labels": CatMetric(),
            "val_predicted": CatMetric(),
            "val_labels": CatMetric(),
            "test_predicted": CatMetric(),
            "test_labels": CatMetric(),
        }

        if label_vocab is None:
            assert label_path is not None
            dialect_classes = open(label_path).read().splitlines()
            ordered_classes = {c: len(dialect_classes) - i for i, c in enumerate(dialect_classes)}
            self.label_vocab = build_vocab(ordered_classes)
        else:
            self.label_vocab = label_vocab

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.save_path = save_path
        if device == "cuda":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, target_ids, related_ids, subrelated_ids):
        attn_tar = target_ids > 0
        _, target = self.bert_model(
            input_ids=target_ids, attention_mask=attn_tar, return_dict=False
        )
        attn_rel = related_ids > 0
        _, related = self.bert_model(
            input_ids=related_ids, attention_mask=attn_rel, return_dict=False
        )
        attn_sub = subrelated_ids > 0
        _, subrelated = self.bert_model(
            input_ids=subrelated_ids, attention_mask=attn_sub, return_dict=False
        )

        target_out = self.dropout(target)
        related_out = self.dropout(related)
        subrelated_out = self.dropout(subrelated)
        out_target = self.fcc_tar(target_out)
        out_related = self.fcc_rel(related_out)
        out_subrelated = self.fcc_sub(subrelated_out)
        res = (
            self.weights[0] * out_target
            + self.weights[1] * out_related
            + self.weights[2] * out_subrelated
        )
        return res

    def step(self, batch, batch_idx, stage="train"):
        ids, rels, subs, labels = batch

        output = self.forward(ids, rels, subs)
        predicted = output.argmax(dim=1)
        labels = labels.view(-1)
        loss = self.loss_fn(output, labels)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.accumulators[f"{stage}_predicted"].update(predicted)
        self.accumulators[f"{stage}_labels"].update(labels)

        return {"loss": loss, "predicted": predicted, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        ids, rels, subs, labels = batch
        output = self(ids, rels, subs)
        predicted = output.argmax(dim=1)
        return (predicted, labels)


    def tokenize_column(self, col):
        return self.tokenizer.encode_plus(
            col,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )['input_ids'].squeeze()

    def predict(self, columns, related_cols=None, subrelated_cols=None):
        tokens = []
        for c in columns:
            tokenized_text = self.tokenize_column(c)
            tokens.append(tokenized_text)
        ids = torch.stack(tokens).to(self.device)

        if related_cols is not None:
            for c in related_cols:
                tokenized_text = self.tokenize_column(c)
                tokens.append(tokenized_text)
            related_ids = torch.stack(tokens).to(self.device)
        else:
            related_ids = torch.zeros_like(ids).to(self.device)

        if subrelated_cols is not None:
            for c in subrelated_cols:
                tokenized_text = self.tokenize_column(c)
                tokens.append(tokenized_text)
            subrelated_ids = torch.stack(tokens).to(self.device)
        else:
            subrelated_ids = torch.zeros_like(ids).to(self.device)

        with torch.no_grad():
            output = self.forward(ids, related_ids, subrelated_ids)
        headers = self.label_vocab.lookup_tokens(output.argmax(dim=1).cpu().numpy())
        return headers

    def epoch_end(self, stage="train"):
        self.weighted_f1.reset()
        self.macro_f1.reset()

        predicted = self.accumulators[f"{stage}_predicted"].compute().cpu().numpy()
        labels = self.accumulators[f"{stage}_labels"].compute().cpu().numpy()

        w_f1 = sk_f1_score(labels, predicted, average="weighted")
        m_f1 = sk_f1_score(labels, predicted, average="macro")

        self.log(
            f"{stage}_w_f1_epoch",
            w_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_m_f1_epoch",
            m_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for acc in ["predicted", "labels"]:
            self.accumulators[f"{stage}_{acc}"].reset()

    def on_train_epoch_end(self):
        self.epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        return self.epoch_end(stage="val")

    def on_test_epoch_end(self) -> None:
        return self.epoch_end(stage="test")

    def save_weights(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        parent_folder = os.path.dirname(save_path)
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        print("Saving model in " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_weights(self, load_path=None):
        if load_path is None:
            load_path = self.save_path
        if os.path.isfile(load_path):
            print("Restoring model from " + load_path)
            wdict = torch.load(load_path)
            self.load_state_dict(wdict)
        else:
            print("Base model not found in " + load_path)

    def configure_optimizers(self):
        weight_decay = 1e-2
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.optimizer_lr
        )
        return optimizer
