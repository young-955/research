# %%
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torch
import numpy as np
import time
from torchsummary import summary
from torchstat import stat
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import grad_scaler as grad_scaler

def eval(model, data, loss_func):
    model.eval()
    loss_list = []
    label_list = []
    pred_list = []

    # for item in eval_data.dataset:
    for item in data:
        data, label = item
        label = label.cuda()
        data = data.cuda()
        pred = model(data)
        pred = torch.softmax(pred, 1)
        loss = loss_func(pred.cpu(), label)
        loss_list.append(loss.detach().cpu().item())
        label_list += list(label.cpu().numpy())
        pred_list += list(torch.argmax(pred.cpu(), 1).numpy())

    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    loss = np.mean(loss_list)
    acc = np.sum(pred_list == label_list) / np.size(label_list)

    model.train()
    return loss, acc

data_transform = transforms.Compose([
    # transforms.Resize([128, 128]),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def slim_BN_l1(model, s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))

def infer_test(model):
    test_dataset = datasets.CIFAR10(root='../../dataset/', train=False, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cost = 0
    count = 0
    model.cuda()
    model.eval()
    for item in test_loader:
        count += 1
        data, label = item
        data = data.cuda()
        st = time.time()
        pred = model(data)
        cost += time.time() - st
    print(f'total cost:{cost}, total batch data:{count}, per batch cost:{cost/count}')
    return

class Model_param():
    def __init__(self, batch_size, epoch, learn_rate, logging_steps, eval_steps, loss_func, optimzer) -> None:
        # Resnet params
        self.batch_size = batch_size
        self.epoch = epoch
        self.learn_rate = learn_rate
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.loss_func = loss_func
        self.optimizer = optimizer

'''
slim:      是否剪枝
slim_l1_s: BN层l1正则化参数
'''
def train(model, mparam: Model_param, slim=False, slim_l1_s = 0.0001):
    train_dataset = datasets.CIFAR10(root='../../dataset/', train=True, transform=data_transform)
    test_dataset = datasets.CIFAR10(root='../../dataset/', train=False, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=mparam.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=mparam.batch_size, shuffle=False)

    model.cuda()
    model.train()

    count =0
    cur_loss = 0
    max_acc = 0
    for e in range(mparam.epoch):
        cur_count = 0
        for data in train_loader:
            img, label = data
            img = img.cuda()
            label = label.cuda()

            pred = model(img)
            loss = mparam.loss_func(pred, label)
            mparam.optimizer.zero_grad()
            loss.backward()
            #是否剪枝
            if slim:
                slim_BN_l1(model, slim_l1_s)
            mparam.optimizer.step()

            cur_loss += loss.item()
            count += 1
            cur_count += 1
            if count % mparam.logging_steps == 0:
                print(f'epoch {e}, count {cur_count}, cur loss {cur_loss}')
                cur_loss = 0
            if count % mparam.eval_steps == 0:
                ev_loss, ev_acc = eval(model, test_loader, mparam.loss_func)
                if ev_acc > max_acc:
                    max_acc = ev_acc
                    torch.save(model, f'cifa10_{ev_acc}.pth')

                print(f'epoch {e}, count {cur_count}, eval loss {ev_loss}, eval accurate {ev_acc}')

# 加载resnet
# Resnet params
batch_size = 128
epoch = 20
learn_rate = 0.0256
momentum = 0.875
weight_decay = 0.00125
logging_steps = 100
eval_steps = 500

loss_func = nn.CrossEntropyLoss()
# model = torchvision.models.resnet50(pretrained=False, num_classes=10)
model = torchvision.models.vgg11(pretrained=False, num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.875, weight_decay=0.00125)

mparam = Model_param(batch_size, epoch, learn_rate, logging_steps, eval_steps, loss_func, optimizer)

train(model, mparam)

# %%
