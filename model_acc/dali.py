from nvidia.dali.backend import TensorCPU
from nvidia.dali.backend import TensorGPU

import torch
import numpy as np

# 导入数据，numpy可以给CPU，torch.cuda可以传给GPU
def get_data_demo():
    a = np.array([1,2])
    b = torch.Tensor(a)

    c = TensorCPU(a)
    d = TensorGPU(b.cuda())
    print(d)

# 预处理
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from torchvision import transforms 
from PIL import Image

img = Image.open('../../data/1.jpeg')
tensor_img = transforms.ToTensor()(img)

dali_img = TensorGPU(tensor_img.cuda())
