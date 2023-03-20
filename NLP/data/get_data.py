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

train_x, val_x, train_y, val_y = text_vecs[:190000], text_vecs[190000:], labels[:190000], labels[190000:]

train_data = humordata(train_x, train_y)
val_data = humordata(val_x, val_y)

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                            batch_size = 1,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_data,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 8)

# %%
import sys
sys.path.append('../')
from series_model.self_lstm import *
from train.train import *

model = LSTM(3, 100, 256, 2)

train(model, train_loader, val_loader)
# %%
