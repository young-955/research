from nvidia.dali.backend import TensorCPU
from nvidia.dali.backend import TensorGPU

import torch
import numpy as np

a = np.array([1,2])
b = torch.Tensor(a)

c = TensorCPU(a)

d = TensorGPU(b.cuda())
print(d)