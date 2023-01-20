from torchsummary import summary
from torchstat import stat
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time

class eval():
    def __init__(self, model, input_size = (1,3,224,224)) -> None:
        self.model = model
        self.model.eval()
        self.loss_func = nn.CrossEntropyLoss()
        data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        test_dataset = datasets.CIFAR10(root='../../dataset/cifar10/', train=False, transform=data_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        self.input = torch.randn(input_size, device='cuda')

    def eval_model(self):
        print('summary:')
        summary(self.model.cuda(), input_size=(3, 32, 32))
        print('stat:')
        stat(self.model.cpu(), (3, 32, 32))

    def profiling(self):
        self.model(self.input)
        avt = 0
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True, with_modules=True) as prof:
            for _ in range(100):
                t1 = time.time()
                self.model(self.input)
                t2 = time.time()
                avt += t2 - t1
        print(f'time cost: {avt / 100}')
        print(prof.key_averages())

    def eval_acc(self, model, data, loss_func):
        self.model.cuda()
        loss_list = []
        label_list = []
        pred_list = []

        # for item in eval_data.dataset:
        for item in data:
            data, label = item
            data = data.cuda()
            pred = self.model(data)
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
