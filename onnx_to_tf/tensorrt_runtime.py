import engine as eng
import inference as inf
import tensorrt as trt
import util
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import cv2

input_file_path = "/home/gorkem/Documents/acceleration/tf_tft_env/lazy.jpg"
serialized_plan_f32 = "/home/gorkem/Documents/acceleration/onnx_to_tf/models/mobilenet.plan"

height = 224
width = 224


class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        pass # Your custom logging implementation here

#logger = MyLogger()
logger = trt.Logger(trt.Logger.VERBOSE)
trt_runtime = trt.Runtime(logger)


image = cv2.imread(input_file_path)
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0
im = np.expand_dims(image, axis=0)

engine = eng.load_engine(trt_runtime, serialized_plan_f32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)

fpsses = []
out = None
with tqdm(total=10000) as pbar:
    for i in range(10000):
        start_time = time.time()
        out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, height, width)
        fps = 1/(time.time() - start_time)
        fpsses.append(fps)
        pbar.update(1)
        
print("AVERAGE FPS: ", np.mean(fpsses))
if np.argmax(out) == 511:
    print("Predicted: CAR")

print("Done ... ")