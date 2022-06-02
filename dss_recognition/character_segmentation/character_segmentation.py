import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from dss_recognition.line_segmentation import dataloader_task1 as ds


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# this function trims white from the left and the right of the input image
def trim_top_bottom(image):
    zero_row_ind, _ = np.where(image == 0)

    first = min(zero_row_ind)
    last = max(zero_row_ind)

    if first < 0: first = 0
    if last > len(image[0]): last = len(image[0])

    return image[first:last, :]


# this function trims white from the left and the right of the input image
def trim_sides(image):
    _, zero_col_ind = np.where(image == 0)

    first = min(zero_col_ind)
    last = max(zero_col_ind)

    if first < 0: first = 0
    if last > len(image[0]): last = len(image[0])

    return image[:, first:last]


# this function takes the locations of the binarized valleys and splits the image on these points
def split_image_to_peaks(image, peaks, widths):
    characters = []

    for i in range(len(peaks)):
        temp = image[:, (peaks[i] - widths[i] / 2).astype(int):(peaks[i] + widths[i] / 2).astype(int)]
        characters.append(temp)

        # cv2.imshow("split", temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return characters


# this function updates the lists representing the binary representation of the histogram
def update_binary_histogram(binary, widths, peaks, left, right):
    new_width = (peaks[right] + widths[right] / 2) - (peaks[left] - widths[left] / 2)
    new_peak = peaks[left] - widths[left] / 2 + new_width / 2

    binary = np.delete(binary, np.arange(left, right + 1))
    widths = np.delete(widths, np.arange(left, right + 1))
    peaks = np.delete(peaks, np.arange(left, right + 1))

    binary = np.insert(binary, left, 1)
    widths = np.insert(widths, left, new_width)
    peaks = np.insert(peaks, left, new_peak)

    return binary, widths, peaks


# case 1: join successive zero's
# case 2: join zero between ones to the closest one
def binary_joining_rule(peaks, widths, binary):
    i = 1

    # handle first 0 in sequence
    if binary[0] == 0 and len(binary) > 1:
        if binary[1] == 0:
            first_zero = 0
        else:
            binary, widths, peaks = update_binary_histogram(binary, widths, peaks, 0, 1)
            first_zero = None
    else:
        first_zero = None

    while i < len(binary) - 1:
        left_zero = True if first_zero else False
        right_zero = True if binary[i + 1] == 0 else False

        # only joining happens with peaks labeled as 0
        if binary[i] == 0:
            if not left_zero and right_zero:
                first_zero = i
            # last zero in sequence
            elif left_zero and not right_zero:
                binary, widths, peaks = update_binary_histogram(binary, widths, peaks, first_zero, i)
                i = first_zero
                first_zero = None
            # zero between two ones
            elif not left_zero and not right_zero:
                # left peak is closer
                if (peaks[i] - widths[i] / 2) - (peaks[i-1] + widths[i-1] / 2) < \
                        (peaks[i+1] - widths[i+1] / 2) - (peaks[i] + widths[i] / 2):
                    binary, widths, peaks = update_binary_histogram(binary, widths, peaks, i-1, i)
                    i -= 1
                # right peak is closer
                else:
                    binary, widths, peaks = update_binary_histogram(binary, widths, peaks, i, i + 1)

                first_zero = None
        i += 1

    # handle last index if last 0 in sequence
    if i < len(binary) and binary[i] == 0 and binary[i-1] == 0:
        _, widths, peaks = update_binary_histogram(binary, widths, peaks, first_zero, i)
    # handle last index if 0 preceded by a 1
    elif i < len(binary) and binary[i] == 0:
        binary, widths, peaks = update_binary_histogram(binary, widths, peaks, i - 1, i)

    return peaks, widths


# this function binarizes peaks and joins peaks based on relative with according to research by
# Rajput, Jayanthi and Sreedevi (2019)
def segment_characters(peaks, widths, image):
    k = 0.7
    max_gap = 20
    binary = []
    split_locations = []
    characters = []

    # compute average width
    width_avg = np.mean(widths)

    # label peaks
    for width in widths:
        if width < k * width_avg:
            binary.append(0)
        else:
            binary.append(1)

    # split labeled array based on maximum distance between two characters to avoid joining over long distance
    for i in range(len(binary) - 1):
        if ((peaks[i + 1] - widths[i + 1] / 2) - (peaks[i] + widths[i] / 2)) > max_gap:
            split_locations.append(i)

    # segment all characters
    if split_locations:
        start = 0
        end = len(binary)

        for i in split_locations:
            new_peaks, new_widths = binary_joining_rule(peaks[start:i + 1], widths[start:i + 1], binary[start:i + 1])
            characters.append(split_image_to_peaks(image, new_peaks, new_widths))
            start = i + 1

        new_peaks, new_widths = binary_joining_rule(peaks[start:end], widths[start:end], binary[start:end])
        characters.append(split_image_to_peaks(image, new_peaks, new_widths))
    else:
        new_peaks, new_widths = binary_joining_rule(peaks, widths, binary)
        characters.append(split_image_to_peaks(image, new_peaks, new_widths))

    return characters


# this function creates a histogram from the input image and determines the splits based on the binarized histogram
def apply_histogram_segmentation(image, plot=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = trim_sides(image)
    image = trim_top_bottom(image)
    im_transp = image.T
    bins = []

    for i in range(len(im_transp)):
        bins.append(np.sum(im_transp[i] == 0))

    # Plot binary graph TUNE FACTOR FOR DIFFERENT THRESHOLDING
    binarized = (bins > 0.9 * np.mean(bins)).astype(np.int_)
    binarized = np.insert(binarized, 0, 0)
    binarized = np.append(binarized, 0)

    if plot:
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout(h_pad=2)

        # show original image
        ax[0].set_title(' image ')
        plt.set_cmap('gray')
        ax[0].imshow(image)

        # Plot pixel graph
        ax[1].set_title(' histogram ')
        ax[1].stairs(bins, fill=False)
        ax[1].set_xlim([0, len(bins)])

        ax[2].set_title(' binarized ')
        ax[2].stairs(binarized, fill=False, color="orange")
        binary_valleys, _ = sp.find_peaks(np.multiply(binarized, -1))
        ax[2].vlines(x=binary_valleys, ymin=0, ymax=1, colors="red")
        ax[2].set_xlim([0, len(binarized)])

        plt.show()

    binary_peaks, _ = sp.find_peaks(binarized)
    widths = sp.peak_widths(binarized, binary_peaks, rel_height=1)

    # TODO: WIDTH IS ONE TOO LARGE FIND A WAY TO DO THIS OTHER THAN -1

    # characters = split_image_to_peaks(image, binary_peaks, widths)
    characters = segment_characters(binary_peaks, widths[0] - 1, image)

    # return right to left
    # return [i[::-1] for i in characters[::-1]]

    # return left to right
    return characters


# this is the main function loading from the crops folder
def character_segmentation():
    segmented_images = []
    images = ds.read_cropped_lines()

    for image in images:
        segmented_lines = []
        for line in image:
            image = cv2.imread(line)
            segmented_lines.append(apply_histogram_segmentation(image))
        segmented_images.append(segmented_lines)

    return segmented_images


if __name__ == "__main__":
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_1.jpg')
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_6.jpg')
    file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_2_crop_3.jpg')
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_16_crop_9.jpg')
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_18_crop_12.jpg')
    words = apply_histogram_segmentation(cv2.imread(file), True)
    for word in words:
        for char in word:
            cv2.imshow("split", char)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
