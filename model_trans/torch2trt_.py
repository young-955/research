# %%
import time
import torch
from torch2trt import torch2trt

model = torch.load('../models/prune_model/vgg16_BN_slimed_cifar10_acc_824.pth').eval().cuda()

test_data = torch.rand((1, 3, 224, 224)).cuda()
# a = 0
# f = 100
# for _ in range(f):
#     t1 = time.time()
#     model(test_data)
#     t2 = time.time()
#     a += t2 - t1
# print(a / f)

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [test_data])
y = model(test_data)
y_trt = model_trt(test_data)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
# %%
# import os
# os.environ["PATH"].split(os.path.pathsep)
# %%
