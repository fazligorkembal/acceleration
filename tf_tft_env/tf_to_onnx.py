import os
import tensorflow as tf
import numpy as np
import onnxruntime as rt
import tf2onnx
import time
import cv2



model = tf.keras.models.load_model("/home/gorkem/Documents/acceleration/tf_tft_env/models/resnet50")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "/home/gorkem/Documents/acceleration/tf_tft_env/models/resnet50.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]