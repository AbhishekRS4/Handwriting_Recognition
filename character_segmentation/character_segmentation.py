import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def trim(image):
    pass


def get_histogram(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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
    # maxima = np.array(bins)[peaks.astype(int)]
    # max_height = max(bins)
    # plt.plot(peaks, max_height, "x")
    # plt.vlines(x=peaks, ymin=0, ymax=max_height, colors="red")

    # Plot binary graph
    binarized = (bins > np.mean(bins)).astype(np.int_)

    ax[2].set_title(' binarized ')
    ax[2].stairs(binarized, fill=False, color="orange")
    binary_valleys, _ = sp.find_peaks(np.multiply(binarized, -1))
    ax[2].vlines(x=binary_valleys, ymin=0, ymax=1, colors="red")
    ax[2].set_xlim([0, len(binarized)])

    plt.show()


if __name__ == "__main__":
    file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_8.jpg')
    get_histogram(file)
