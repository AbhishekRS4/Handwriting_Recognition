import argparse
import cv2
import os
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from character_segmentation import character_segmentation as cs
from line_segmentation import blob_line_mask_extraction as bme
from line_segmentation.LineExtraction2 import run_matlab_code as rmc
from line_segmentation import horizontal_hist_projection as hhp
import line_segmentation.dataloader_task1 as dl
from character_recognition import load_alexnet
import numpy as np

hebrew_characters = {
    0:'alef',
    1: 'ayin',
    2: 'bet',
    3: 'dalet',
    4: 'gimel',
    5: 'he',
    6: 'het',
    7: 'kaf',
    8: 'kaf-final',
    9: 'lamed',
    10: 'mem',
    11: 'mem-medial',
    12: 'nun-final',
    13: 'nun-medial',
    14: 'pe',
    15: 'pe-final',
    16: 'qof',
    17: 'resh',
    18: 'samekh',
    19: 'shin',
    20: 'taw',
    21: 'tet',
    22: 'tsadi-final',
    23: 'tsadi-medial',
    24: 'waw',
    25: 'yod',
    26: 'zayin'
    }

hebrew_decimal_codes = {
    "alef": 1488,
    "ayin": 1506,
    "bet": 1489,
    "dalet": 1491,
    "gimel": 1490,
    "he": 1492,
    "het": 1495,
    "kaf": 1499,
    "kaf-final": 1498,
    "lamed": 1500,
    "mem": 1501,
    "mem-medial": 1502,
    "nun-final": 1503,
    "nun-medial": 1504,
    "pe": 1508,
    "pe-final": 1507,
    "qof": 1511,
    "resh": 1512,
    "samekh": 1505,
    "shin": 1513,
    "taw": 1514,
    "tet": 1496,
    "tsadi-final": 1509,
    "tsadi-medial": 1510,
    "waw": 1493,
    "yod": 1497,
    "zayin": 1494,
}

def start_dss_recognize(FLAGS):
    model = load_alexnet.model()
    expected_width = 64
    expected_height = 64

    list_image_files = dl.read_binarized_images(FLAGS.dir_images)

    if not os.path.isdir(FLAGS.dir_save_predictions):
        os.makedirs(FLAGS.dir_save_predictions)

    for image_file in list_image_files:
        print(image_file)
        file_handler = open(os.path.join(FLAGS.dir_save_predictions, os.path.basename(image_file)+".txt"), encoding="utf-8", mode="w")
        separated_masks = rmc.extract_masks(image_file)
        if FLAGS.line_segment_method == 'blob':
            line_images = bme.extract_lines(image_file, separated_masks)
        else:
            line_images = hhp.extract_lines(image_file)

        for line_image in line_images:
            segmented_chars = cs.apply_histogram_segmentation(line_image)
            for word in segmented_chars:
                for char in word:
                    i = 0
                    original_image = cv2.cvtColor(char, cv2.COLOR_GRAY2RGB)
                    image = cv2.resize(original_image, (expected_width, expected_height), cv2.INTER_LINEAR)
                    image = image.astype("float") / 255.0
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    prediction_char = model.predict(image)
                    position_char = np.argmax(prediction_char)

                    pred_char = hebrew_characters[position_char]
                    hebrew = hebrew_decimal_codes[pred_char]

                    file_handler.write(chr(hebrew))

                    #if Viterbi > x:
                    #prob.append(hebrew_characters) #np.argmax
                    i = i + 1
                file_handler.write(" ")
            file_handler.write("\n")
        file_handler.close()
    return

def main():
    # dir_images = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/img/"
    dir_images = None
    dir_save_predictions = "results"
    line_segment_method = "blob"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full path to directory containing dead sea scroll images")
    parser.add_argument("--dir_save_predictions", default=dir_save_predictions,
        type=str, help="directory to save the predictions")
    parser.add_argument("--line_segment_method", default=line_segment_method,
        type=str, help="Method for line segmentation, type 'blob' for blob-line method, type 'hhp' for horizontal histogram projection")

    FLAGS, unparsed = parser.parse_known_args()
    start_dss_recognize(FLAGS)
    return

if __name__ == "__main__":
    main()
