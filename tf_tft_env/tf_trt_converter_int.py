from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import copy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf

input_saved_model_dir = "/home/gorkem/Documents/acceleration/tf_tft_env/models/resnet50"

batched_size = 8
batched_input = np.zeros((batched_size, 224, 224, 3), dtype=np.float32)

for i in range(batched_size):
    img_path = "/home/gorkem/Documents/acceleration/tf_tft_env/lazy.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batched_input[i, :] = x
batched_input = tf.constant(batched_input)

print("Converting to TF-TRT Int8")
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.INT8,
    max_workspace_size_bytes=8000000000,
    use_calibration=True
)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    conversion_params=conversion_params
)

def calibration_input_fn():
    yield (batched_input,)

converter.convert(calibration_input_fn=calibration_input_fn)
converter.save("/home/gorkem/Documents/acceleration/tf_tft_env/models/tf_trt_resnet50_int")

