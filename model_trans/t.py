import tensorrt as trt
import torch
import time
from trt_infer import allocate_buffers, do_inference_v2
import torch

test_data = torch.rand((1, 3, 224, 224)).numpy()

with open('trt_int8.plan', 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine, context)
inputs[0].host = test_data
a = 0
f = 100
for _ in range(f):
    t1 = time.time()
    trt_outs = do_inference_v2(context, bindings, inputs, outputs, stream)
    t2 = time.time()
    a += t2 - t1
print(a / f)

import time
time.sleep(5)