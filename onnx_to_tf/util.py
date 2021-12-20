import numpy as np
from PIL import Image
import tensorrt as trt
import skimage.transform

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

MEAN = (71.60167789, 82.09696889, 72.30508881)
CLASSES = 20

HEIGHT = 512
WIDTH = 1024

def sub_mean_chw(data):
   data = data.transpose((1, 2, 0))  # CHW -> HWC
   data -= np.array(MEAN)  # Broadcast subtract
   data = data.transpose((2, 0, 1))  # HWC -> CHW
   return data

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

def color_map(output):
   output = output.reshape(CLASSES, HEIGHT, WIDTH)
   out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
   for x in range(WIDTH):
       for y in range(HEIGHT):

           if (np.argmax(output[:, y, x] )== 19):
               out_col[y,x] = (0, 0, 0)
   return out_col  