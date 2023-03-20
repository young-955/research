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
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from torchvision import transforms 
from PIL import Image
import nvidia.dali.fn as fn

@pipeline_def
def ppline():
    dali_img, label = fn.readers.file(files='../../data/1.jpeg')
    dali_img = fn.decoders.image(dali_img, output_type=types.RGB)
    dali_img = fn.normalize(dali_img.gpu())
    dali_img = fn.transpose(dali_img, perm=[2, 0, 1])
    return dali_img, label


pp = ppline(batch_size=1, num_threads=3, device_id=0)
pp.build()

# img, _ = pp.run()
# print(type(img[0]))
# print(img[0].shape())
# # help(img[0])
# print(img[0])
import torchvision
model = torchvision.models.vgg11(pretrained=False, num_classes=10)

model.cuda()

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
loader = DALIClassificationIterator(pp)
for d in loader:
    print(d[0]['data'].shape)
    res = model(d[0]['data'])
    print(res)


# import matplotlib.pyplot as plt
# plt.imshow(img[0])
# plt.show()