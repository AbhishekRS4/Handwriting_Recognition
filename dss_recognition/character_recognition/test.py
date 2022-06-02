import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2

hebrew_characters = [
    'Alef': 0,
    'Ayin': 1,
    'Bet': 2,
    'Dalet': 3,
    'Gimel': 4,
    'He': 5,
    'Het': 6,
    'Kaf': 7,
    'Kaf-final': 8,
    'Lamed': 9,
    'Mem': 10,
    'Mem-medial': 11,
    'Nun-final': 12,
    'Nun-medial': 13,
    'Pe': 14,
    'Pe-final': 15,
    'Qof': 16,
    'Resh': 17,
    'Samekh': 18,
    'Shin': 19,
    'Taw': 20,
    'Tet': 21,
    'Tsadi-final': 22,
    'Tsadi-medial': 23,
    'Waw': 24,
    'Yod': 25,
    'Zayin': 26
    ]



expected_width = 64
expected_height = 64
saved_model = 'alexnet.h5'
original_image = cv2.imread("image")
# Preprocessing the image
image = cv2.resize(original_image, (expected_width, expected_height))

