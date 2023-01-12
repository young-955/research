# bn层和conv层融合

# %%
import torch
import torch.nn as nn
import sys
sys.path.append('../')
from eval.eval import eval

bn_conv_model = torch.load('../../models/prune_model/vgg16_FC_BN_slimed_cifar10_acc_8211.pth')

def conv_bn_merge(conv, bn):
    kernel = conv.weight.data
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight.data
    beta = bn.bias.data
    eps = bn.eps
    std = torch.sqrt(running_var+eps)
    t = (gamma/std).view(-1,1,1,1).expand(kernel.size())
    kernel = kernel*t
    bias = beta - running_mean*gamma/std

    conv.weight.data = kernel.clone()
    conv.bias.data = bias.clone()
    return conv

# %%
last_v = None
layers = []
for v in bn_conv_model.features.children():
    if isinstance(v, nn.Conv2d):
        last_v = v
        continue
    if isinstance(last_v, nn.Conv2d) and isinstance(v, nn.BatchNorm2d):
        new_conv = conv_bn_merge(last_v, v)
        last_v = v
        layers.append(new_conv)
        continue
    layers.append(v)
    last_v = v
new_model = nn.Sequential(*layers)
bn_conv_model.features = new_model
# %%
e_tool = eval(bn_conv_model)
e_tool.eval_model()


# %%
torch.save('../../models/conv_bn_combine/vgg16_FC_BN_slimed_CONV_BN_COMBINE_cifar10.pth')