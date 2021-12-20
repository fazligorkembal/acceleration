import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import onnxruntime as rt
import tf2onnx
import time
import cv2

img_path = '/home/gorkem/Documents/acceleration/images/ade20k.jpg'

model = MobileNet(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model.save('/home/gorkem/Documents/acceleration/models/mobilenet.h5')

image = cv2.imread(img_path)
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "/home/gorkem/Documents/acceleration/models/mobilenet.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]