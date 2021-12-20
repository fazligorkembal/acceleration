import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import onnxruntime as rt
import tf2onnx
import time

img_path = '/home/gorkem/Documents/acceleration/images/ade20k.jpg'



model = tf.keras.models.load_model('/home/gorkem/Documents/acceleration/models/semantic_segmentation.hdf5')
start_time = time.time()
print("Tensorflow Runtime Took:", time.time() - start_time)
spec = (tf.TensorSpec((None, 3, 512, 1024), tf.float32, name="input"),)
output_path = "/home/gorkem/Documents/acceleration/models/semantic_segmentation.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

print("Output Names:", output_names)

#Run The ONNX Model
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
start_time = time.time()



