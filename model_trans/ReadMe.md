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
    2.

### 验证结果

方法 | 耗时 |
--- | --- | ---
onnx自带trt | 0.0293 |
onnx转trt | 0.0047 |
