import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    def __init__(self, verbose=False):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 8 * (2 ** 30)  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None
    
    def create_network(self, onnx_path):
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        with open(onnx_path, 'rb') as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to parse ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        print(self.network)
        print("Network Created ...")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print(input.name, input.dtype, input.shape)
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
            print(output.name, output.dtype, output.shape)
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size
    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=25000,
                      calib_batch_size=8, calib_preprocessor=None):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        """

        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("Platform does not support fast fp16")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            self.config.set_flag(trt.BuilderFlag.INT8)
            sys.exit("INT8 not supported yet")
        with self.builder.build_engine(self.network, self.config) as f:
            log.info("Serializing Engine to {}".format(engine_path))
            
        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def main(verbose, onnx, engine_path, precision_type):
    """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
    """
    
    
    builder = EngineBuilder(verbose=verbose)
    builder.create_network(onnx)
    builder.create_engine(engine_path, precision_type)





if __name__ == "__main__":
    verbose=True
    onnx_path = "/home/gorkem/Documents/acceleration/jetson_xavier_nx_python/models/centernet.onnx"
    engine_path = onnx_path.replace(".onnx", ".plan")
    precision_type = "fp16"

    main(verbose, onnx_path, engine_path, precision_type)

