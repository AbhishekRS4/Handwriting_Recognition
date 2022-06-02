import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2

hebrew_characters = [
    'Alef','Ayin','Bet','Dalet','Gimel','He','Het','Kaf','Kaf-final','Lamed','Mem','Mem-medial','Nun-final','Nun-medial','Pe','Pe-final','Qof',
    'Resh','Samekh','Shin','Taw','Tet','Tsadi-final','Tsadi-medial','Waw','Yod','Zayin'
    ]



expected_width = 64
expected_height = 64
saved_model = 'alexnet.h5'
original_image = cv2.imread("image")
# Preprocessing the image
image = cv2.resize(original_image, (expected_width, expected_height))

