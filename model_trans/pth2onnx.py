# %%
# torch2onnx
import torch

model = torch.load('../models/prune_model/vgg16_BN_slimed_cifar10_acc_824.pth')
test_data = torch.rand((1, 3, 224, 224)).cuda()
torch.onnx.export(model, args=test_data, f='test.onnx', verbose=True, input_names=['input'], output_names=['output'],)
# %%
# pth time
import torch
import time
a = 0
f = 100
model = torch.load('../models/prune_model/vgg16_BN_slimed_cifar10_acc_824.pth').eval().cuda()
test_data = torch.rand((1, 3, 224, 224)).cuda()
for _ in range(f):
    t1 = time.time()
    model(test_data)
    t2 = time.time()
    a += t2 - t1
print(a / f)
# %%
# onnx time
import onnxruntime
import torch
import time
test_data = torch.rand((1, 3, 224, 224)).cuda()
onnx_session = onnxruntime.InferenceSession('test.onnx')
a = 0
f = 100
for _ in range(f):
    t1 = time.time()
    onnx_session.run(['output'], {'input': test_data.cpu().numpy()})
    t2 = time.time()
    a += t2 - t1
print(a / f)
# %%
