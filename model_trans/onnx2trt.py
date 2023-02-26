# %%
# 使用onnx中的trt核 
import onnxruntime
import torch
import time

# test_data = torch.rand((1, 3, 224, 224)).cuda()

# onnx_session = onnxruntime.InferenceSession('test.onnx',
#         providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

# a = 0
# f = 100
# for _ in range(f):
#     t1 = time.time()
#     onnx_session.run(['output'], {'input': test_data.cpu().numpy()})
#     t2 = time.time()
#     a += t2 - t1
# print(a / f)

# %%
# onnx 转 trt
# windows 下会缺zlib库，百度下一个即可
import tensorrt as trt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()

trt_calibrator = trt.IInt8Calibrator()

def build_engine(onnx_file_path, engine_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.max_workspace_size = 1 << 32  # 4GB
        builder.max_batch_size = 1
        # 是否fp16
        # config.flags |= 1 << int(trt.BuilderFlag.FP16)
        # config.flags |= 1 << int(trt.BuilderFlag.INT8)
        # builder.int8_mode = True

        
        '''
        int8
        IBuilderConfig: 
        int8_calibrator: :class:`IInt8Calibrator` Int8 Calibration interface. 
        The calibrator is to minimize the information loss during the INT8 quantization process.

        flags: :class:`int` The build mode flags to turn on builder options for this network. 
        The flags are listed in the BuilderFlags enum. The flags set configuration options to build the network. 
        This should be in integer consisting of one or more :class:`BuilderFlag` s, 
        combined via binary OR. For example, ``1 << BuilderFlag.FP16 | 1 << BuilderFlag.DEBUG``.
        '''

        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)

build_engine('test.onnx', 'trt_int8.plan')

