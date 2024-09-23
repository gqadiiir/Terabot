import keras
import tensorflow as tf
from deepface import DeepFace

print("keras version:", keras.__version__)
print("tensorflow version:", tf.__version__)


result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")