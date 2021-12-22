import onnx
import tensorrt as trt

def build_engine(
    onnx_path,
    seq_len=192,
    max_seq_len=256,
    batch_size=8,
    max_batch_size=64,
    trt_fp16=True,
    verbose=True,
    max_workspace_size=None,
    encoder=True,
    ):   

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = max_batch_size

    with open(onnx_path, 'rb') as model_fh:
        model = model_fh.read()

    model_onnx = onnx.load_model_from_string(model)
    input_feats = model_onnx.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    input_name = model_onnx.graph.input[0].name

    if trt_fp16:
        config_flags = 1 << int(trt.BuilderFlag.FP16)  # | 1 << int(trt.BuilderFlag.STRICT_TYPES)
    else:
        config_flags = 0
    builder.max_workspace_size = max_workspace_size if max_workspace_size else (4 * 1024 * 1024 * 1024)

    config = builder.create_builder_config()
    config.flags = config_flags

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(1, input_feats, seq_len),
        opt=(batch_size, input_feats, seq_len),
        max=(max_batch_size, input_feats, max_seq_len),
    )
    config.add_optimization_profile(profile)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        parsed = parser.parse(model)
        print("Parsing returned ", parsed)
        return builder.build_engine(network, config=config)

if __name__ == "__main__":
    onnx_path = "/home/gorkem/Documents/acceleration/onnx_to_tf/models/mobilenet.onnx"
    seq_len = 192
    max_seq_len = 256
    max_batch_size=64
    batch_size=8
    mode_fp16 = True

    engine = build_engine(onnx_path, seq_len=seq_len, max_seq_len=max_seq_len, batch_size=batch_size, max_batch_size=max_batch_size, trt_fp16=mode_fp16)
    print("Done ... ")