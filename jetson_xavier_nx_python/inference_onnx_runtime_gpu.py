import onnxruntime
import numpy as np
import cv2
from tqdm import tqdm
import onnx

image_path = "/home/gorkem/Documents/acceleration/tf_tft_env/lazy.jpg"
model_onnx_path = "/home/gorkem/Documents/acceleration/jetson_xavier_nx_python/models/test.onnx"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.resize(image, (224, 224))
#image = image / 255.0
image = np.expand_dims(image, axis=0)
image = image.astype(np.uint8)

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]


session = onnxruntime.InferenceSession(model_onnx_path, providers=providers)
print(session.get_providers())

output_name = session.get_outputs()[0].name

# get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
input_name = session.get_inputs()[0].name

# inference run using image_data as the input to the model 
print(onnxruntime.get_device())
with tqdm(total=100000) as pbar:
    for i in range(100000):
        detections = session.run([output_name], {input_name: image})[0]
        pbar.update(1)
        