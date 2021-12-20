import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import util
from tqdm import tqdm
import time
import cv2

model_path = "/home/gorkem/Documents/acceleration/models/mobilenet.h5"
input_file_path = "/home/gorkem/Documents/acceleration/images/ade20k.jpg"
height = 224
width = 224

model = tf.keras.models.load_model(model_path)

image = cv2.imread(input_file_path)
image = cv2.resize(image, (224, 224))
image = image.astype(np.float16) / 255.0
im = np.expand_dims(image, axis=0)

fpsses = []
out = None
with tqdm(total=1000) as pbar:
    for i in range(1000):
        start_time = time.time()
        out = model.predict(im)
        fps = 1/(time.time() - start_time)
        fpsses.append(fps)
        pbar.update(1)

print("AVERAGE FPS: ", np.mean(fpsses))
if np.argmax(out) == 511:
    print("Predicted: CAR")