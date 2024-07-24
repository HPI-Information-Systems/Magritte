import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        if len(np.unique(labels)) > 2:
            self.labels = [l / 128 if l<128 else 1. for l in labels]
        self.labels = torch.tensor(list(self.labels))#.unsqueeze(0)
        self.labels = self.labels.float()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        label = self.labels[index]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        return embedding, label