import os
import cv2
import argparse
import numpy as np
import scipy.signal as sp

from line_segmentation import LineExtraction2.run_matlab_code as rmc
from line_segmentation.blob_line_mask_extractor import get_segment_crop, apply_mask
from character_segmentation.character_segmentation import trim_sides, segment_characters

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

def apply_histogram_char_segmentation(line_image):
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    _, line_image = cv2.threshold(line_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    line_image = trim_sides(line_image)
    im_transp = line_image.T
    bins = []

    for i in range(len(im_transp)):
        bins.append(np.sum(im_transp[i] == 0))

    # Plot binary graph TUNE FACTOR FOR DIFFERENT THRESHOLDING
    binarized = (bins > 0.9 * np.mean(bins)).astype(np.int_)
    binary_peaks, _ = sp.find_peaks(binarized)
    widths = sp.peak_widths(binarized, binary_peaks, rel_height=1)

    # characters = split_image_to_peaks(image, binary_peaks, widths)
    segmented_characters = segment_characters(binary_peaks, widths[0], line_image)

    return segmented_characters

def extract_lines(image_file, masks, idx):
    image = cv2.imread(image_file)

    # Remove background using bitwise-and operation
    i = 1
    line_images = []
    for mask in masks:
        crop_im, crop_mask = get_segment_crop(image, mask=mask)
        line_image = apply_mask(crop_im, crop_mask)
        line_images.append(line_image)
    return line_images

def start_dss_recognize(FLAGS):
    list_image_files = os.listdir(FLAGS.dir_images)
    list_image_files = [os.path.join(FLAGS.dir_images, f) for f in list_image_files]

    for image_file in list_image_files:
        file_handler = open(os.path.join(FLAGS.dir_save_predictions, image_file+".txt"), encoding="utf-8", mode="w")
        separated_masks = rmc.extract_masks(file)
        line_images = extract_lines(image_file, separated_masks, i)

        for line_image in line_images:
            segmented_chars = apply_histogram_char_segmentation(line_image)

            for segmented_char in segmented_chars:
                #predict char using network
                #label = network.predict(segmented_char)
                # handle left to right or right to left chars
                file_handler.write(chr(hebrew_decimal_codes[hebrew_label_encodings[label]]))
            file_handler.write("\n")
        file_handler.close()
    return

def main():
    dir_images = "/home/abhishek/Desktop/RUG/hw_recognition/IAM-data/img/"
    dir_save_predictions = "results"

    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full path to directory containing dead sea scroll images")
    parser.add_argument("--dir_save_predictions", default=dir_save_predictions,
        type=str, help="directory to save the predictions")

    FLAGS, unparsed = parser.parse_known_args()
    start_dss_recognize(FLAGS)
    return

if __name__ == "__main__":
    main()
