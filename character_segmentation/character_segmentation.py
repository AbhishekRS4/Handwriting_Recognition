import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# this function trims white from the left and the right of the input image
def trim_sides(image):
    _, zero_col_ind = np.where(image == 0)

    first = min(zero_col_ind)
    last = max(zero_col_ind)

    return image[:, first:last]


# this function takes the locations of the binarized valleys and splits the image on these points
def split_characters(image, peaks, widths):
    characters = []

    for i in range(len(peaks)):
        temp = image[:, (peaks[i] - widths[i] / 2).astype(int):(peaks[i] + widths[i] / 2).astype(int)]
        characters.append(temp)

        cv2.imshow("split", temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return characters


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
def temp_binary_joining_rule(peaks, widths, binary):
    # TODO: incorporate max distance between peaks to join
    i = 1
    first_zero = 0 if binary[0] == 0 else None

    while i < len(binary) - 1:
        left_zero = True if first_zero else False
        right_zero = True if binary[i + 1] == 0 else False

        # only joining happens with peaks labeled as 0
        if binary[i] == 0:
            # first zero in sequence
            if not left_zero and right_zero:
                first_zero = i
            # last zero in sequence
            elif left_zero and not right_zero:
                binary, widths, peaks = update_binary_histogram(binary, widths, peaks, first_zero, i)

                # new_width = (peaks[i] + widths[i] / 2) - (peaks[first_zero] - widths[first_zero] / 2)
                # new_peak = peaks[first_zero] - widths[first_zero] / 2 + new_width / 2
                #
                # binary = np.delete(binary, np.arange(first_zero, i+1))
                # widths = np.delete(widths, np.arange(first_zero, i+1))
                # peaks = np.delete(peaks, np.arange(first_zero, i+1))
                #
                # binary = np.insert(binary, first_zero, 1)
                # widths = np.insert(widths, first_zero, new_width)
                # peaks = np.insert(peaks, first_zero, new_peak)

                i = first_zero
                first_zero = None
            # zero between two ones
            elif not left_zero and not right_zero:
                # left peak is closer
                if peaks[i-1] + widths[i-1] / 2 < peaks[i+1] - widths[i+1] / 2:
                    binary, widths, peaks = update_binary_histogram(binary, widths, peaks, i-1, i)
                    # new_width = (peaks[i] + widths[i] / 2) - (peaks[i-1] - widths[i-1] / 2)
                    # new_peak = peaks[i-1] - widths[i-1] / 2 + new_width / 2
                    #
                    # binary = np.delete(binary, np.arange(i-1, i+1))
                    # widths = np.delete(widths, np.arange(i-1, i+1))
                    # peaks = np.delete(peaks, np.arange(i-1, i+1))
                    #
                    # binary = np.insert(binary, i-1, 1)
                    # widths = np.insert(widths, i-1, new_width)
                    # peaks = np.insert(peaks, i-1, new_peak)

                    i -= 1
                # right peak is closer
                else:
                    binary, widths, peaks = update_binary_histogram(binary, widths, peaks, i, i + 1)
                    # new_width = (peaks[i+1] + widths[i+1] / 2) - (peaks[i] - widths[i] / 2)
                    # new_peak = peaks[i] - widths[i] / 2 + new_width / 2
                    #
                    # binary = np.delete(binary, np.arange(i, i+2))
                    # widths = np.delete(widths, np.arange(i, i+2))
                    # peaks = np.delete(peaks, np.arange(i, i+2))
                    #
                    # binary = np.insert(binary, i, 1)
                    # widths = np.insert(widths, i, new_width)
                    # peaks = np.insert(peaks, i, new_peak)

                first_zero = None
        i += 1

    # handle last index if last 0 in sequence
    if binary[i] == 0 and binary[i-1] == 0:
        _, widths, peaks = update_binary_histogram(binary, widths, peaks, first_zero, i)
        # new_width = (peaks[i] + widths[i] / 2) - (peaks[first_zero] - widths[first_zero] / 2)
        # new_peak = peaks[first_zero] - widths[first_zero] / 2 + new_width / 2
        #
        # widths = np.delete(widths, np.arange(first_zero, i+1))
        # peaks = np.delete(peaks, np.arange(first_zero, i+1))
        #
        # widths = np.insert(widths, first_zero, new_width)
        # peaks = np.insert(peaks, first_zero, new_peak)
    # handle last index if 0 preceded by a 1
    elif binary[i] == 0:
        binary, widths, peaks = update_binary_histogram(binary, widths, peaks, i - 1, i)
        # new_width = (peaks[i] + widths[i] / 2) - (peaks[i - 1] - widths[i - 1] / 2)
        # new_peak = peaks[i - 1] - widths[i - 1] / 2 + new_width / 2
        #
        # widths = np.delete(widths, np.arange(i - 1, i + 1))
        # peaks = np.delete(peaks, np.arange(i - 1, i + 1))
        #
        # widths = np.insert(widths, i - 1, new_width)
        # peaks = np.insert(peaks, i - 1, new_peak)

    return peaks, widths


# this function binarizes peaks and joins peaks based on relative with according to research by
# Rajput, Jayanthi and Sreedevi (2019)
def segment_characters(peaks, widths, image):
    k = 0.7

    # compute average width
    width_avg = np.mean(widths)

    binary = []

    # label peaks
    for width in widths:
        if width < k * width_avg:
            binary.append(0)
        else:
            binary.append(1)

    new_peaks, new_widths = temp_binary_joining_rule(peaks, widths, binary)

    characters = split_characters(image, new_peaks, new_widths)

    return characters


# this function creates a histogram from the input image and determines the splits based on the binarized histogram
def apply_histogram_segmentation(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = trim_sides(image)
    im_transp = image.T
    bins = []

    for i in range(len(im_transp)):
        bins.append(np.sum(im_transp[i] == 0))

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
    inv_data = np.multiply(bins, -1)
    peaks, _ = sp.find_peaks(inv_data)

    # Plot binary graph
    binarized = (bins > np.mean(bins)).astype(np.int_)
    # binarized = (bins > np.mean(bins)/2).astype(np.int_)

    ax[2].set_title(' binarized ')
    ax[2].stairs(binarized, fill=False, color="orange")
    binary_valleys, _ = sp.find_peaks(np.multiply(binarized, -1))
    ax[2].vlines(x=binary_valleys, ymin=0, ymax=1, colors="red")
    ax[2].set_xlim([0, len(binarized)])

    plt.show()

    # characters = split_characters_at_valley(image, binary_valleys)

    binary_peaks, _ = sp.find_peaks(binarized)
    widths = sp.peak_widths(binarized, binary_peaks, rel_height=1)

    characters = segment_characters(binary_peaks, widths[0], image)


if __name__ == "__main__":
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_8.jpg')
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_6.jpg')
    file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_2_crop_10.jpg')
    apply_histogram_segmentation(file)
