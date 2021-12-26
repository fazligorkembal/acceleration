import os
import sys
import argparse

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
from image_batcher import ImageBatcher
from tqdm import tqdm 

class TensorrtInference:

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        self.inputs = []
        self.outputs = []
        self.allocations = []
        print("Engine:", self.engine)
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': 1,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),   
                'shape': list(shape),
                'allocation': allocation
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) >0


    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        return self.outputs[0]['shape'], self.outputs[0]['dtype']
    
    def infer(self, batch, top=1):
        output = np.zeros(*self.output_spec())
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[0]['allocation'])

        classes = np.argmax(output, axis=1)
    
        scores = np.max(output, axis=1)
        top = min(top, output.shape[1])
        top_classes = np.flip(np.argsort(output, axis=1), axis=1)[:, 0:top]
        top_scores = np.flip(np.sort(output, axis=1), axis=1)[:, 0:top]

        return classes, scores, [top_classes, top_scores]

def main(engine_path, input, preprocess_input, separator, top):
    trt_infer = TensorrtInference(engine_path)
    batcher = ImageBatcher(input, *trt_infer.input_spec(), preprocessor=preprocess_input)
    for batch, images in batcher.get_batch():
        with tqdm(total=100000) as pbar:
            for i in range(100000):
                if top == 1:
                    classes, scores, tops = trt_infer.infer(batch)
                    print(classes, scores, tops)
                pbar.update(1)


if __name__ == "__main__":
    engine_path = "/home/gorkem/Documents/acceleration/jetson_xavier_nx_python/models/mobilenet_onnx_float32_engine_int8.plan"
    image_path = "/home/gorkem/Documents/acceleration/jetson_xavier_nx_python/images/l1.jpg"
    top = 1
    seperator = "\t"
    preprocessor = "V2"
    main(engine_path, image_path, preprocessor, seperator, top)
