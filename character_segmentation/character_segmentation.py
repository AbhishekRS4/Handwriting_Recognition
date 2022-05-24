import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_histogram(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_transp = image.T
    bins = []

    for i in range(len(im_transp)):
        bins.append(np.sum(im_transp[i] == 0))

    fig = plt.figure()

    # show original image
    fig.add_subplot(211)
    plt.title(' image ')
    plt.set_cmap('gray')
    plt.imshow(image)

    fig.add_subplot(212)

    # Plot graph
    plt.title('histogram ')
    plt.stairs(bins, fill=True)

    peaks, _ = sp.find_peaks(bins)
    maxima = np.array(bins)[peaks.astype(int)]

    plt.plot(peaks, maxima, "x")
    plt.vlines(x=peaks, ymin=0, ymax=maxima, colors="red")

    plt.show()


if __name__ == "__main__":
    file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_8.jpg')
    get_histogram(file)
