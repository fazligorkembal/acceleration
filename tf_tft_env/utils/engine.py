import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape = [1,300,300,3]):
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = 1 << 32
        print("Has platform fast16", builder.platform_has_fast_fp16)
        print("Has platform fastINT8", builder.platform_has_fast_int8)
        

        with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)

def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine