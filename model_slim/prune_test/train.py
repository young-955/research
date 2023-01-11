# %%
import sys
sys.path.append('../../paper_reproduction/VGG')
import pandas as pd
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from vgg_reproduction import *
import torchvision

from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import pickle

# init
class img_dataset(Dataset):
    def __init__(
        self,
        data,
        label
    ):
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def evaluate(model, device, loss_func, eval_data):
    model.eval()
    loss_list = []
    label_list = []
    pred_list = []

    # for item in eval_data.dataset:
    for item in eval_data:
        data, label = item
        label = torch.Tensor([list(label_dict[l.item()]) for l in label])
        data = torch.permute(data, [0, 3, 1, 2])
        data = ((data / 255) - 0.5) / 0.5
        data = data.to(device)
        pred = model(data)
        pred = torch.softmax(pred, 1)
        loss = loss_func(pred.cpu(), label)
        loss_list.append(loss.detach().cpu().item())
        label_list += list(torch.argmax(label, 1).numpy())
        pred_list += list(torch.argmax(pred.cpu(), 1).numpy())

    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    loss = np.mean(loss_list)
    acc = np.sum(pred_list == label_list) / np.size(label_list)

    model.train()
    return loss, acc


# %%
# train_path = "data/train"
# t = pd.read_csv('data/Training_set.csv')
# pic_label = list(set(t['label'].values))
# onehot_label = F.one_hot(torch.arange(0, len(pic_label)))
# label_dict = {k: v for k, v in zip(pic_label, onehot_label)}

# # build train dataset
# train_data = []
# train_label = t['label'].values
# for img_name in t['filename'].values:
#     img_path = os.path.join(train_path, img_name)
#     img = cv2.resize(cv2.imread(img_path), (224, 224))
#     train_data.append(img)

# data = img_dataset(train_data, train_label)


# %%
train_data = pickle.load(open('data/CIFAR10_train_data.pkl', 'rb'))
train_label = pickle.load(open('data/CIFAR10_train_label.pkl', 'rb'))
test_data = pickle.load(open('data/CIFAR10_test_data.pkl', 'rb'))
test_label = pickle.load(open('data/CIFAR10_test_label.pkl', 'rb'))

onehot_label = F.one_hot(torch.arange(0, 10))
label_dict = {k: v for k, v in zip(np.arange(0, 10), onehot_label)}


# %%
# build myself model
# vgg_net = VGG(model_type='D', input_size=(32, 32), output_size=len(label_dict.keys()))
# vgg_net = VGG(model_type='A', input_size=(32, 32), output_size=len(label_dict.keys()))
# vgg_net.load_state_dict(torch.load('./trained_model/vgg11-acc41-20221113.pth'), strict=False)

# use pretrained model
vgg_net = torchvision.models.vgg16(pretrained=False)
vgg_net.load_state_dict(torch.load('./trained_model/vgg-pretrained.pth'), strict=False)
# freeze pretrained layers
# for param in vgg_net.features.parameters():
#     param.requires_grad = False
vgg_net = nn.Sequential(
    vgg_net,
    nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(0.5, inplace=False),
        nn.Linear(1000, 10)
    )
)

batch_size = 1
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vgg_net.parameters(), lr=1e-4, weight_decay=0.01)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
# train_dataloader = DataLoader(data, batch_size=batch_size,
#                             num_workers=0, drop_last=True)
# train_size = int(0.8 * len(train_dataloader))
# test_size = len(train_dataloader) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(train_dataloader, [train_size, test_size])
train_dataloader = DataLoader(img_dataset(train_data, train_label), batch_size=batch_size,
                            num_workers=0, drop_last=True)
test_dataloader = DataLoader(img_dataset(test_data, test_label), batch_size=batch_size,
                            num_workers=0, drop_last=True)
# %%
# train model
t_loss = 0.0
global_steps = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_epoch = 3
logging_steps = 100
eval_steps = 1000
model_save_dir = './trained_model/'
best_metric = 0.0
best_steps = 1

vgg_net.to(device)
loss_func.to(device)
vgg_net.train()
for epoch in range(train_epoch):
    # for step, item in enumerate(train_dataset.dataset):
    for step, item in enumerate(train_dataloader):
        data, label = item
        label = torch.Tensor([list(label_dict[l.item()]) for l in label])
        data = torch.permute(data, [0, 3, 1, 2])
        data = ((data / 255) - 0.5) / 0.5
        data = data.to(device)
        pred = vgg_net(data)
        pred = torch.softmax(pred, 1)
        loss = loss_func(pred.cpu(), label)
        
        optimizer.zero_grad()
        loss.backward()
        t_loss += loss.detach()

        optimizer.step()
        # scheduler.step()

        if global_steps % logging_steps == 0:
            print(f'Training: Epoch {epoch + 1}/{train_epoch} - Step {step + 1} - Loss {t_loss}')
            t_loss = 0.0

        if (global_steps + 1) % eval_steps == 0:
            # eval_loss, acc = evaluate(vgg_net, device, loss_func, test_dataset)
            eval_loss, acc = evaluate(vgg_net, device, loss_func, test_dataloader)
            print(f'Evaluation: Epoch {epoch + 1}/{train_epoch} - Step {global_steps + 1} - Loss {eval_loss} - Accuracy {acc}')

            if acc > best_metric:
                best_metric = acc
                best_steps = global_steps
                
                torch.save(vgg_net.state_dict(), f'{model_save_dir}checkpoint-{best_steps}-acc{acc}.pth')

        global_steps += 1

# %%

import torch
import torchvision
vgg_net = torchvision.models.vgg16(pretrained=False)
vgg_net.load_state_dict(torch.load('./trained_model/vgg-pretrained.pth'), strict=False)

# %%
import torch
import torchvision
vgg_net = torchvision.models.vgg16(pretrained=True)
# %%
vgg_net
# %%
torch.save(vgg_net, './trained_model/torch_vgg16.pth')
# %%
list(vgg_net.children())
# %%
