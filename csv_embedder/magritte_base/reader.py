"""
This script contains the CSV reader of MAGRITTE base
 read() takes a path to a csv file and calls instance after splitting it into max_rows
 text_to_instance generate instances that are sequences of tokens, indexed with a vocabulary, for each row
"""
import csv
import logging
import sys
from typing import Dict

import chardet
import numpy as np
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.token_indexers import TokenIndexer
from overrides import overrides

from csv_embedder.pattern_tokenizer import PatternTokenizer

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


@DatasetReader.register("magritte_base_reader")
class MagritteBaseReader(DatasetReader):
    def __init__(self,
                 tokenizer: PatternTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_rows: int = 10,
                 max_len: int = 32,
                 **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_rows = max_rows
        self.max_len = max_len

    @overrides
    def _read(self, file_path, *args, **kwargs):
        rawfile = open(file_path, "rb").read()
        try:
            file = rawfile.decode("utf-8")
        except Exception as e:
            encoding = chardet.detect(rawfile)["encoding"]
            file = rawfile.decode(encoding)

        lines = file.splitlines()
        yield self.text_to_instance(lines)

    @overrides
    def text_to_instance(self, lines, *args, **kwargs) -> Instance:
        fields = {}
        tokens = []
        token_type_ids = []
        for l in lines:
            tok = self._tokenizer.tokenize(l)
            # Zero Paddings
            if len(tok) < self.max_len - 1:
                n_pad = self.max_len - len(tok)
                tok.extend([Token(self.pad_token)] * n_pad)

            tokens += [[Token(self.cls_token)] + tok[:self.max_len - 1]]

        if len(tokens) < self.max_rows:
            pad_dim = max(self.max_rows - len(tokens), 0)
            indices = np.random.choice(range(len(tokens)), pad_dim)
            tokens += [tokens[i] for i in indices]

        # Random padding
        fields["input_ids"] = ListField([TextField(tok) for tok in tokens])
        fields["token_type_ids"] = ListField([LabelField(t, skip_indexing=True) for t in token_type_ids])

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["input_ids"].token_indexers = self._token_indexers