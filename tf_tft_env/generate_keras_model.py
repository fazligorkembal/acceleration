import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet


model = MobileNet(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model.save('/home/gorkem/Documents/acceleration/tf_tft_env/models/resnet50')
