import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch 
from torch import nn, optim
from train.eval import *

epoch = 20
learn_rate = 1e-2
logging_steps = 100
eval_steps = 1000
# BN层l1正则化参数
slim_l1_s = 0.0001

'''
    slim: 是否剪枝
'''
def train_func(model, train_loader, test_loader, w2v_model, slim=False):
    model.cuda()
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    count =0
    cur_loss = 0
    for e in range(epoch):
        cur_count = 0
        print(1)
        for data in train_loader:
            data, label = data
            print(2)

            text_vec = []
            for char in data:
                try:
                    vec = w2v_model[char]
                    text_vec.append(vec)
                except:
                    pass
            text_vec = torch.Tensor(text_vec).cuda()

            label = label.cuda()
            pred = model(text_vec)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            if slim:
                slim_BN_l1(model, slim_l1_s)
            optimizer.step()
            cur_loss += loss.item()
            count += 1
            cur_count += 1
            print(cur_count)
            if count % logging_steps == 0:
                print(f'epoch {e}, count {cur_count}, cur loss {cur_loss}')
                cur_loss = 0
            if count % eval_steps == 0:
                ev_loss, ev_acc = eval(model, test_loader, loss_func)
                print(f'epoch {e}, count {cur_count}, eval loss {ev_loss}, eval accurate {ev_acc}')
