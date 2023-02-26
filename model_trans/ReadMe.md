# 模型转换


## 1.pth转onnx

### 方法

    torch直接转onnx

### 验证结果

以VGG16为例，pth转onnx后，推理性能提升一倍

模型类型 | 耗时 |
--- | --- | ---
pth | 0.0735 |
onnx | 0.0424 |


## 2.onnx转tensorrt


### 方法

    1.直接使用onnx中自带的trt核
    2.使用onnx转tensorrt，用tensorrt加载推理
    3.使用torch2trt模块

### 验证结果

方法 | 耗时 | 显存占用
--- | --- | --- | --- |
onnx自带trt | 0.0293 | |
onnx转trt(CPU) | 0.002164 | |
trt fp16(CPU) | 0.00139 | |
trt int8(CPU) | 0.0011868 |
torch2trt | | |
