import argparse
import cv2
import os
from character_segmentation import character_segmentation as cs
from line_segmentation import blob_line_mask_extraction as bme
from line_segmentation.LineExtraction2 import run_matlab_code as rmc
import dataloader_task1 as dl
from character_recognition import load_alexnet
import numpy as np
import keras

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

hebrew_label_encodings = {

}

def start_dss_recognize(FLAGS):
    # list_image_files = os.listdir(FLAGS.dir_images)
    # list_image_files = [os.path.join(FLAGS.dir_images, f) for f in list_image_files]
    list_image_files = dl.read_binarized_images(FLAGS.dir_images)

    for image_file in list_image_files:
        # file_handler = open(os.path.join(FLAGS.dir_save_predictions, image_file+".txt"), encoding="utf-8", mode="w")
        separated_masks = rmc.extract_masks(image_file)
        line_images = bme.extract_lines(image_file, separated_masks)
        print(line_images)

        for line_image in line_images:
            segmented_chars = cs.apply_histogram_segmentation(line_image)
            print(segmented_chars)

            for word in segmented_chars:
                print(word)
                load_alexnet()
                expected_width = 64
                expected_height = 64

                segmented_characters_path = 'testing_model'
                for subdir, dirs, files in os.walk(segmented_characters_path):
                        predicted_char = []
                        i = 0
                        for f in files:
                            original_image = cv2.imread(f"{subdir}/{f}")
                            image = cv2.resize(original_image, (expected_width, expected_height), cv2.INTER_LINEAR)
                            image = image.astype("float") / 255.0
                            image = img_to_array(image)
                            image = np.expand_dims(image, axis=0)
                            prediction_char = model.predict(image)
                            position_char = np.argmax(prediction_char)
                            print(hebrew_characters[position_char])
                            #if Viterbi > x:
                            #prob.append(hebrew_characters) #np.argmax
                            i = i + 1
                        #need to reverse, since  hebrew characters are read from right to left
                        #predicted_char.reverse()
                
                    
        #     file_handler.write("\n")
        # file_handler.close()
    return

def main():
    # dir_images = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/img/"
    dir_images = None
    dir_save_predictions = "results"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full path to directory containing dead sea scroll images")
    parser.add_argument("--dir_save_predictions", default=dir_save_predictions,
        type=str, help="directory to save the predictions")

    FLAGS, unparsed = parser.parse_known_args()
    start_dss_recognize(FLAGS)
    return

if __name__ == "__main__":
    main()
