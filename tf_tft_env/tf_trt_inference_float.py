import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time
import cv2
from tqdm import tqdm

input_saved_model_path = "/home/gorkem/Documents/acceleration/tf_tft_env/models/tf_trt_mobilenet"
model = tf.saved_model.load(input_saved_model_path, tags=[tag_constants.SERVING])
signature_keys = list(model.signatures.keys())
print(signature_keys)

infer = model.signatures['serving_default']
print(infer.structured_outputs)

img = image.load_img("/home/gorkem/Documents/acceleration/tf_tft_env/lazy.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
x = tf.constant(x)
#x = cv2.imread("/home/gorkem/Documents/acceleration/tf_tft_env/lazy.jpg")
#x = tf.expand_dims(x, axis=0)

fpses = []
with tqdm(total=100000) as pbar:
    for i in range(100000):
        start_time = time.time()
        labeling = infer(x)
        fps = 1.0 / (time.time() - start_time)
        fpses.append(fps)
        pbar.update(1)
print("Average FPS:", np.mean(fpses))
