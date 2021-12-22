from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import copy

input_saved_model_dir = "/home/gorkem/Documents/acceleration/tf_tft_env/models/mobilenet"

params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
precision_mode = trt.TrtPrecisionMode.FP16
params = params._replace(
    precision_mode=precision_mode,
    #max_workspace_size_bytes=2 << 32,
    max_workspace_size_bytes=8000000000,
    maximum_cached_engines=100,
    minimum_segment_size=3,
    allow_build_at_runtime=True
)

converter = trt.TrtGraphConverterV2(input_saved_model_dir, conversion_params=params)
converter.convert()
converter.save("/home/gorkem/Documents/acceleration/tf_tft_env/models/tf_trt_mobilenet")
print("Done ... ")