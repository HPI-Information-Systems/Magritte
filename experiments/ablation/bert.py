from experiments.ablation.model import EmbeddingsModel
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import pandas as pd
import csv
#from turl import TURL_embedding_generator
import os

def prepare_sentence_list(table: str, max_lines: int) -> list[str]:
    """Utility function to convert tables in a format that can be processed by bert

    Args:
        table (str): path to a .csv file
        max_lines (int): truncate input after this numer of lines

    Returns:
        list[str]: list containing an entry for every row inside the input file, entry are the concatenation of the content of the cells
    """
    table = pd.read_csv(table, on_bad_lines='skip', encoding_errors='ignore', quoting=csv.QUOTE_NONE, dtype=str, lineterminator='\n', header=None)
    out = []
    for r in range(min(table.shape[0], max_lines)):
        row = []
        for c in range(table.shape[1]):
            row.append(str(table.iloc[r].iloc[c]))
        out.append(" ".join(row))
    return out

def prepare_sentence_list_bert(table: str, max_lines: int) -> list[str]:
    rows = []
    counter = 0
    with open(table, 'rt', errors='ignore') as t:
        for r in t.readlines():
            if counter >= max_lines:
                break
            rows.append(r)
            counter += 1
    return rows

class Bert_Embedding_Generator:

    def __init__(self,
                 max_lines = 128,
                 output_format = 'mean',
                 output_hidden_states: bool=False, 
                 model: str='bert') -> None:
        """The class init method

        Args:
            output_hidden_states (bool, optional): NA. Defaults to False.
            model (str, optional): name of the model to use. Defaults to 'bert'.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=output_hidden_states).to(self.device).eval()
        elif model == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base') 
            self.model = RobertaModel.from_pretrained('FacebookAI/roberta-base', output_hidden_states=output_hidden_states).to(self.device).eval()
        else:
            raise Exception('Model not supported')
        self.max_lines = max_lines
        self.output_format = output_format

    def encode(self, l: list[str]) -> list:
        """Perform an encoding operation necessary to generate the embeddings

        Args:
            l (str): the str to encode

        Returns:
            list: a list of encodings
        """    
        try:
            return self.tokenizer(l, padding=True, truncation=True, max_length=128, add_special_tokens=True)
        except:
            l = ['']
            return self.tokenizer(l, padding=True, truncation=True, max_length=128, add_special_tokens=True)


    def __call__(self, sentence: str, strategy: str='CLS') -> torch.Tensor:
        """The class call method, it manages the emebdding generation

        Args:
            sentence (str): sentence to embed
            strategy (str, optional): do not modify. Defaults to 'CLS'.

        Returns:
            torch.Tensor: the embedding of the sentence
        """
        enc = self.encode(sentence)
        enc = {k:torch.LongTensor(v).to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        hidden_states = out['last_hidden_state']

        if strategy == 'CLS':
            sentence_embedding = hidden_states[:,0]

        elif strategy == 'average':
            sentence_embedding = torch.mean(hidden_states, dim=1)
        
        if len(sentence) == 1:
            return sentence_embedding.squeeze(0)
        else:
            return sentence_embedding

    def embed(self, file: str) -> dict[str, torch.Tensor]:
        """Given the path to a file generate a table embedding

        Args:
            file (str): path to the file to embed

        Raises:
            NotImplementedError: thrown if a now implemented output format is requested

        Returns:
            dict[str, torch.Tensor]: returns a dictionary with the entry "file_emebdding" that contains the file embeddings
        """
        sentences = prepare_sentence_list_bert(file, self.max_lines)
        if self.output_format=='concat' and len(sentences) < self.max_lines:
            sentences += ['' for _ in range(self.max_lines-len(sentences))]
        embeddings = self.__call__(sentences)
        if self.output_format == 'mean':
            try:
                embeddings.shape[1]
                embeddings = torch.mean(embeddings, dim=0)
            except:
                pass
        elif self.output_format == 'concat':
            embeddings = embeddings.reshape(-1)
        else:
            raise NotImplementedError
        
        return {"file_embedding":embeddings}


class Table_embedder(EmbeddingsModel):
    """
        Utility class that provides a common interface to generate table embeddings using various embedding models
    """
    def __init__(self, 
                 embedding_model: str='bert', 
                 output_format: str='mean', 
                 max_lines: int=128,
                 sampling_size: int=10,
                 checkpoints_directory: str = 'results/ablation_regression/') -> None:
        """__init__ method

        Args:
            embedding_model (str, optional): embedding model to use. Defaults to 'bert'.
            output_format (str, optional): strtegy to follow while aggregating the row embeddings. Defaults to 'mean'.
            max_lines (int, optional): truncate all of the lines in the input file after this number. Defaults to 128.

        Raises:
            NotImplementedError: thrown if a not implemented embedding model is required
        """
        self.checkpoints_directory = checkpoints_directory
        if embedding_model == 'bert':
            self.model = Bert_Embedding_Generator(model='bert', max_lines=max_lines, output_format=output_format)
        elif embedding_model == 'roberta':
            self.model = Bert_Embedding_Generator(model='roberta', max_lines=max_lines, output_format=output_format)
        elif embedding_model == 'turl':
            pass
        else:
            raise NotImplementedError

if __name__ == '__main__':
    # sentence = ['']
    # erb = Bert_Embedding_Generator('roberta')
    # e = erb(sentence=sentence)
    # tt = Table_embedder(embedding_model='roberta', output_format='mean')
    # emb = tt.embed('/home/francesco.pugnaloni/test_docker/test_table.csv')
    # emb = tt.embed('/data/survey/csv/Transfer7-tecRep1-d-5,P4&5 (BioRep3).csv')
    # emb = tt.embed('/data/survey/csv/dhsc-spend-over-25000-may-2018.csv')
    # emb = tt.embed('/data/survey/csv/dhsc-spend-over-25000-may-2018.csv')
    # print('gg')
    path = "data/survey/"
    train_dataset = pd.read_csv(path + "train.csv")
    tt = Table_embedder(embedding_model='turl')
    tt.get_embeddings(filenames=train_dataset["filename"], dataset_name='train', model_name='turl')    