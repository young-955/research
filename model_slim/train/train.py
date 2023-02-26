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

from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

def evaluate(model, loss_func, eval_data):
    model.eval()
    loss_list = []
    label_list = []
    pred_list = []

    # for item in eval_data.dataset:
    for item in eval_data:
        data, label = item
        # label = torch.Tensor([list(label_dict[l.item()]) for l in label])
        # data = torch.permute(data, [0, 3, 1, 2])
        # data = ((data / 255) - 0.5) / 0.5
        label = label.cuda()
        data = data.cuda()
        pred = model(data)
        pred = torch.softmax(pred, 1)
        loss = loss_func(pred, label)
        loss_list.append(loss.detach().cpu().item())
        label_list += list(label.cpu().numpy())
        pred_list += list(torch.argmax(pred.cpu(), 1).numpy())

    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    loss = np.mean(loss_list)
    acc = np.sum(pred_list == label_list) / np.size(label_list)

    model.train()
    return loss, acc


# %%
batch_size = 32

data_transform = transforms.Compose([
    # transforms.Resize([128, 128]),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

onehot_label = F.one_hot(torch.arange(0, 10))
label_dict = {k: v for k, v in zip(np.arange(0, 10), onehot_label)}


# %%
# build myself model
# vgg_net = VGG(model_type='D', input_size=(32, 32), output_size=len(label_dict.keys()))
# vgg_net = VGG(model_type='A', input_size=(32, 32), output_size=len(label_dict.keys()))
# vgg_net.load_state_dict(torch.load('./trained_model/vgg11-acc41-20221113.pth'), strict=False)

# use pretrained model
vgg_net = torchvision.models.vgg16(pretrained=True)
# vgg_net.load_state_dict(torch.load('./trained_model/vgg-pretrained.pth'), strict=False)

# freeze some layer
vgg_net.classifier[6]=nn.Linear(4096,10)
for i, param in enumerate(vgg_net.parameters()):
    if i < 16:
        param.requires_grad = False
    else:
        param.requires_grad = True

train_dataset = datasets.CIFAR10(root='../../dataset/cifar10/', train=True, transform=data_transform)
test_dataset = datasets.CIFAR10(root='../../dataset/cifar10/', train=False, transform=data_transform)


loss_func = nn.CrossEntropyLoss()
# use 
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, vgg_net.parameters()), lr=1e-5, weight_decay=0.01)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
# train_dataloader = DataLoader(data, batch_size=batch_size,
#                             num_workers=0, drop_last=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=True)
# %%
# train model
t_loss = 0.0
global_steps = 0
train_epoch = 3
logging_steps = 100
eval_steps = 1000
model_save_dir = './'
best_metric = 0.0
best_steps = 1

vgg_net.cuda()
loss_func.cuda()
vgg_net.train()
for epoch in range(train_epoch):
    for step, item in enumerate(train_dataloader):
        data, label = item
        data = data.cuda()
        label = label.cuda()
        pred = vgg_net(data)
        pred = torch.softmax(pred, 1)
        loss = loss_func(pred, label)
        
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
            eval_loss, acc = evaluate(vgg_net, loss_func, test_dataloader)
            print(f'Evaluation: Epoch {epoch + 1}/{train_epoch} - Step {global_steps + 1} - Loss {eval_loss} - Accuracy {acc}')

            if acc > best_metric:
                best_metric = acc
                best_steps = global_steps
                
                torch.save(vgg_net.state_dict(), f'{model_save_dir}checkpoint-{best_steps}-acc{acc}.pth')

        global_steps += 1

