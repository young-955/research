import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data

from test_model import mymodel
import hiddenlayer as hid


# 模型可视化
mm = mymodel()
m = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
vis_graph = hid.build_graph(mm, torch.zeros([1, 1, 28, 28]))
vis_graph.theme = hid.graph.THEMES["blue"].copy()
vis_graph.save('demo1.png')

