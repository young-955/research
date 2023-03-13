# %%
import pandas as pd
import torch
from torch.utils import data

class humordata(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)
    
from gensim.models import KeyedVectors
w2v_model = KeyedVectors.load('../data/word2vec.model')

dataset = pd.read_csv('dataset.csv')

labels = dataset[['humor']]
labels = [torch.Tensor([0, 1]) if label == True else torch.Tensor([1, 0]) for label in labels.values]

texts = dataset[['text']].values
texts = [text[0].lower().replace('\'', '').replace(',', '').replace('.', '').replace('?', '').replace(':', '').replace(';', '').replace('"', '').split(' ') for text in texts]

# %%
text_vecs = []
for text in texts:
    text_vec = []
    for char in text:
        try:
            vec = w2v_model[char]
            text_vec.append(vec)
        except:
            pass
    text_vecs.append(torch.Tensor(text_vec))

# %%
