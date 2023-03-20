import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch 
from torch import nn, optim
import numpy as np

def eval(model, data, loss_func):
    model.eval()
    loss_list = []
    label_list = []
    pred_list = []

    # for item in eval_data.dataset:
    for item in data:
        data, label = item
        data = data.cuda()
        pred = model(data)
        pred = torch.softmax(pred, 1)
        loss = loss_func(pred.cpu(), label)
        loss_list.append(loss.detach().cpu().item())
        label_list += list(label.numpy())
        pred_list += list(torch.argmax(pred.cpu(), 1).numpy())

    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    loss = np.mean(loss_list)
    acc = np.sum(pred_list == label_list) / np.size(label_list)

    model.train()
    return loss, acc

def slim_BN_l1(model, s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))

