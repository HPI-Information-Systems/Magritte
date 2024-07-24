
from .bert import Table_embedder
from .magritte_models import *
import pickle
import os

class Turl_embedder(EmbeddingsModel):
    def __init__(self, embeddings_path: str='tmp/magritte_ablation/turl_embeddings_tmp/turl_embeddings_final_5_minutes.pkl'): #'/data/magritte_results/local_results_and_turl_embeddings/turl_embeddings/turl_embeddings_final_5_minutes.pkl'):#
        with open(embeddings_path, 'rb') as f:
            self.embedding_dictionary = pickle.load(f)

    def get_embeddings(self, model_name, 
                    filenames, 
                    dataset_name, 
                    filedir="data/survey/csv/"):
        embeddings = []
        for f in tqdm(filenames):
            try:
                embedding = self.embedding_dictionary['embeddings'][f]
            except:
                embedding = torch.zeros(312)
            try:
                if embedding.shape[0] == 1:
                    embedding = embedding.squeeze()
            except:
                pass
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        return embeddings


def EmbeddingsFactory(model_name, checkpoints_directory):
    kwargs = {"checkpoints_directory":checkpoints_directory}
    if model_name == "bert_mean":
        return Table_embedder(embedding_model="bert", output_format="mean", **kwargs)
    elif model_name == "bert_concat":
        return Table_embedder(embedding_model="bert", output_format="concat", **kwargs)
    elif model_name == "roberta_mean":
        return Table_embedder(embedding_model="roberta", output_format="mean", **kwargs)
    elif model_name == "roberta_concat":
        return Table_embedder(embedding_model="roberta", output_format="concat", **kwargs)
    elif model_name == "turl":
        return Turl_embedder()
    elif model_name == "magritte-pt-sfp-cnn":
        return MagrittePT_SFP_CNN(**kwargs)
    elif model_name == "magritte-pt-cnn":
        return MagrittePT_CNN(**kwargs)
    elif model_name == "magritte-pt-sfp":
        return MagrittePT_SFP(**kwargs)
    elif model_name == "magritte-pt":
        return MagrittePT(**kwargs)
    elif model_name == "magritte-wp-sfp-cnn":
        return MagritteWP_SFP_CNN(**kwargs)
    elif model_name == "magritte-wp-cnn":
        return MagritteWP_CNN(**kwargs)
    elif model_name == "magritte-wp-sfp":
        return MagritteWP_SFP(**kwargs)
    elif model_name == "magritte-wp":
        return MagritteWP(**kwargs)
    else:
        raise Exception(f"Model {model_name} not supported")
