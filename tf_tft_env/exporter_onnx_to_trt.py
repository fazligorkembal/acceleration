import onnx
import tensorrt as trt

def build_engine(onnx_path, shape = [1,224,224,3]):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        builder.fp16_mode = True
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine