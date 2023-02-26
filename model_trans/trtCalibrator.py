from tensorrt import IInt8Calibrator
import tensorrt as trt
import pycuda.driver as cuda

class trtCalibrator(IInt8Calibrator):
    def __init__(self,):
        super(trtCalibrator, self).__init__()
        self.batch_size = 32
        self.algorithm = trt.tensorrt.CalibrationAlgoType.ENTROPY_CALIBRATION

    def get_batch(self, names):
        try:
            if self.current_batch >= self.num_batch:
                return None
            assert set(names) == self.names
            # Assume self.batches is a generator that provides batch data.
            data = self.iter.next()
            self.decode_data(data)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            [
                cuda.memcpy_htod(
                    self.host_device_mem_dic[name].device,
                    self.host_device_mem_dic[name].host,
                )
                for name in names
            ]
            self.current_batch += 1
            return [int(self.host_device_mem_dic[name].device) for name in names]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def decode_data(self, data):
        raise NotImplementedError

    def get_batch_size(self):
        return self.batch_size

    def get_algorithm(self):
        return self.algorithm

    def read_calibration_cache(self):
        pass

    def write_calibration_cache(self, cache):
        pass