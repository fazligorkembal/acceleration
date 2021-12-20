import onnxruntime as rt
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
import util
import cv2

sess = rt.InferenceSession('/home/gorkem/Documents/acceleration/models/mobilenet.onnx')

input_file_path = '/home/gorkem/Documents/acceleration/images/ade20k.jpg'

image = cv2.imread(input_file_path)
image = cv2.resize(image, (224, 224))
image = image.astype(np.float16) / 255.0
im = np.expand_dims(image, axis=0)
print(im.shape)

fpses = []
with tqdm(total=1000) as pbar:
    for i in range(1000):
        start_time = time.time()
        preds = sess.run(None, {'input': im})
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fpses.append(fps)
        pbar.update(1)
print("Average Fps:", np.mean(fpses))
