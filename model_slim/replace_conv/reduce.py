# %%
from torchsummary import summary
from torchstat import stat
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time

model = torch.load('../../models/prune_model/vgg16_cifar10_acc_8415.pth')
# %%
type(model.features)
# %%
type(model.features[0].weight)
# %%
for channel_mat in model.features[0].weight:
    for mat in channel_mat:
        print(type(mat))
# %%
