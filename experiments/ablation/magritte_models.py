
from csv_embedder.magritte_pretrain_ae.model import MagrittePretrainingAEReduced
from csv_embedder.magritte_pretrain_rowpair.model import MagrittePretrainingRowPair
from .model import EmbeddingsModel

# torchtext.disable_torchtext_deprecation_warning()
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Some weights of the model checkpoint*")
import json
import sys

sys.path.append(".")
sys.path.append("../../")

from tqdm import tqdm as tqdm
import torch

from csv_embedder.magritte_base.model import MagritteBase
from csv_embedder.pattern_tokenizer import PatternTokenizer

from torchtext.vocab import vocab as build_vocab
from torchtext.transforms import BERTTokenizer, StrToIntTransform, PadTransform


class MagritteAblationModel(EmbeddingsModel):
    config_file = None
    model_class = MagritteBase

    def get_tokenizer(self, config):
        vocab_path = config["vocabulary"]["path"]
        tokens = open(vocab_path).read().splitlines()
        tokens[tokens.index("")] = "\n"
        ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
        token_vocab = build_vocab(ordered_tokens)
        token_vocab.set_default_index(token_vocab["[UNK]"])
        tokenizer = PatternTokenizer(token_vocab=token_vocab, **config["tokenizer"])
        return token_vocab, tokenizer

    def __init__(self, checkpoints_directory):
        self.checkpoints_directory = checkpoints_directory
        with open(self.config_file) as f:
            confstr = f.read()
        config = json.loads(confstr)
        token_vocab, tokenizer = self.get_tokenizer(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (
            self.model_class(
                token_vocab=token_vocab,
                **config["model"],
                tokenizer=tokenizer,
                device=device,
            )
            .to(device)
            .eval()
        )


class MagrittePT_SFP_CNN(MagritteAblationModel):

    config_file = "configs/ablation_eval/magritte_pt_sfp_cnn.json"
    model_class = MagritteBase
    embtype = "file"


class MagrittePT_CNN(MagritteAblationModel):

    config_file: str = "configs/ablation_eval/magritte_pt_cnn.json"
    model_class = MagritteBase
    embtype = "file"

class MagrittePT_SFP(MagritteAblationModel):

    config_file = "configs/ablation_eval/magritte_pt_sfp.json"
    model_class = MagritteBase
    embtype = "row"

class MagrittePT(MagritteAblationModel):

    config_file = "configs/ablation_eval/magritte_pt.json"
    model_class = MagritteBase
    embtype = "row"

class BertFileTokenizer():

    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.tokenizer = BERTTokenizer(vocab_path=vocab_path, do_lower_case=True, return_tokens=False)
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_transform = PadTransform(128, 0)
        self.str_transform = StrToIntTransform()

    def tokenize_file(self, file_path):
        
        with open(file_path, "r", encoding="latin1") as f:
            rows = f.read().splitlines()
        
        if len(rows) < 128:
            rows += [''] * (128 - len(rows))

        row_tokens = []
        for r in rows[:128]:
            if len(r):
                tokens = self.tokenizer(self.cls_token + r)
            else:
                tokens = self.tokenizer(f"{self.cls_token} {self.sep_token}")
            tokens = self.str_transform(tokens)
            tokens = torch.tensor(tokens[:128])
            tokens = self.pad_transform(tokens)
            row_tokens.append(tokens)
            
        out = torch.stack(row_tokens)
        return out
        

class WordPieceAblationModel(MagritteAblationModel):

    def get_tokenizer(self, config):
        vocab_path = config["vocabulary"]["path"]
        tokenizer = BertFileTokenizer(vocab_path=vocab_path)
        tokens = open(vocab_path).read().splitlines()
        if "[PAD]" not in tokens:
            tokens.append("[PAD]")
        if "[CLS]" not in tokens:
            tokens.append("[CLS]")
        ordered_tokens = {t: len(tokens) - i for i, t in enumerate(tokens)}
        token_vocab = build_vocab(ordered_tokens)
        token_vocab.set_default_index(token_vocab["[UNK]"])
        return token_vocab, tokenizer

class MagritteWP_SFP_CNN(WordPieceAblationModel):

    config_file = "configs/ablation_eval/magritte_wp_sfp_cnn.json"
    embtype = "file"

class MagritteWP_SFP(WordPieceAblationModel):

    config_file = "configs/ablation_eval/magritte_wp_sfp.json"
    embtype = "row"
class MagritteWP_CNN(WordPieceAblationModel):

    config_file = "configs/ablation_eval/magritte_wp_cnn.json"
    embtype = "file"

class MagritteWP(WordPieceAblationModel):

    config_file = "configs/ablation_eval/magritte_wp.json"
    embtype = "row"