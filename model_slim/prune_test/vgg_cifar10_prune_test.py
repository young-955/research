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

data_transform = transforms.Compose([
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
    test_dataset = datasets.CIFAR10(root='../dataset/', train=False, transform=data_transform)
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

# params
batch_size = 64
epoch = 20
learn_rate = 1e-2
logging_steps = 100
eval_steps = 1000
# 是否剪枝
slim = True
# BN层l1正则化参数
slim_l1_s = 0.0001

vgg_net = torchvision.models.vgg16_bn(pretrained=False, num_classes=10)

def train(model):
    train_dataset = datasets.CIFAR10(root='../dataset/', train=True, transform=data_transform)
    test_dataset = datasets.CIFAR10(root='../dataset/', train=False, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    count =0
    cur_loss = 0
    for e in range(epoch):
        cur_count = 0
        for data in train_loader:
            img, label = data
            # img = Variable(img)
            img = img.cuda()
            label = label.cuda()
            pred = model(img)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            if slim:
                slim_BN_l1(model, slim_l1_s)
            optimizer.step()
            cur_loss += loss.item()
            count += 1
            cur_count += 1
            if count % logging_steps == 0:
                print(f'epoch {e}, count {cur_count}, cur loss {cur_loss}')
                cur_loss = 0
            if count % eval_steps == 0:
                ev_loss, ev_acc = eval(model, test_loader, loss_func)
                print(f'epoch {e}, count {cur_count}, eval loss {ev_loss}, eval accurate {ev_acc}')
# %%
torch.save(vgg_net, './trained_model/vgg_bnl1_cifar10_acc_8188.pth')
# %%
vgg_net = torch.load('trained_model/vgg_bnl1_cifar10_acc_8188.pth')

prune_rate = 0.7

bn_size = 0
for k,v in vgg_net.named_modules():
    if isinstance(v, nn.BatchNorm2d):
        bn_size += v.weight.data.shape[0]
weights = torch.zeros(bn_size)
indx = 0
for k,v in vgg_net.named_modules():
    if isinstance(v, nn.BatchNorm2d):
        size = v.weight.data.shape[0]
        weights[indx: indx + size] = v.weight.data.abs().clone()
        indx += size

prune_num = int(bn_size * prune_rate)
y, i = torch.sort(weights)
prune_val = y[prune_num]

cfg = []
cfg_mask = []
for k,v in vgg_net.named_modules():
    if isinstance(v, nn.BatchNorm2d):
        weight_cp = v.weight.data.clone()
        mask = weight_cp.abs().gt(prune_val).float().cuda()
        v.weight.data.mul_(mask)
        v.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
    elif isinstance(v, nn.MaxPool2d):
        cfg.append('M')

slim_vgg = torchvision.models.vgg.VGG(torchvision.models.vgg.make_layers(cfg=cfg, batch_norm=True), num_classes=10)

# weight copy
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for m0, m1 in zip(vgg_net.modules(), slim_vgg.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[idx1].clone()
        m1.bias.data = m0.bias.data[idx1].clone()
        m1.running_mean = m0.running_mean[idx1].clone()
        m1.running_var = m0.running_var[idx1].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        w = m0.weight.data[:, idx0, :, :].clone()
        w = w[idx1, :, :, :].clone()
        m1.weight.data = w.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[:, idx0].clone()

test_ipt = torch.autograd.Variable(torch.zeros(1, 3, 32, 32)).cuda()
slim_vgg.cuda()
test_out = torch.flatten(slim_vgg.avgpool(slim_vgg.features(test_ipt)), 1)
slim_vgg.classifier[0] = nn.Linear(test_out.size(1), 1024)
slim_vgg.classifier[3] = nn.Linear(1024, 1024)
slim_vgg.classifier[6] = nn.Linear(1024, 10)

# %%
epoch = 15
slim_vgg.train()
train(slim_vgg)

# %%
torch.save(slim_vgg, './trained_model/vgg_FC_BN_slimed_cifar10_acc_824.pth')
# %%
vgg_net = torch.load('trained_model/vgg_bnl1_cifar10_acc_8188.pth')
bn_slim_vgg = torch.load('trained_model/vgg_BN_slimed_cifar10_acc_824.pth')
fc_bn_slim_vgg = torch.load('trained_model/vgg_FC_BN_slimed_cifar10_acc_8211.pth')

# %%
infer_test(vgg_net)
infer_test(bn_slim_vgg)
infer_test(fc_bn_slim_vgg)
# %%
summary(fc_bn_slim_vgg, input_size=(3, 32, 32))
# %%
stat(fc_bn_slim_vgg.cpu(), (3, 32, 32))

# %%
# pth转onnx
fc_bn_slim_vgg.eval()
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1, 3, 224, 224, device='cuda')
 
torch.onnx.export(fc_bn_slim_vgg, x, 'vgg_FC_BN_slimed_cifar10_acc_8211.onnx', input_names=input_names, output_names=output_names, verbose='True')
