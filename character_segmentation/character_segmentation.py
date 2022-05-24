import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_histogram(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_transp = image.T
    bins = []

    for i in range(len(im_transp)):
        bins.append(np.sum(im_transp[i] == 0))

    print(len(bins))

    fig = plt.figure()

    # show original image
    fig.add_subplot(211)
    plt.title(' image ')
    plt.set_cmap('gray')
    plt.imshow(image)

    fig.add_subplot(212)

    # Plot graph
    plt.title('histogram ')
    plt.bar(range(len(bins)), bins, width=1.0)

    plt.show()


if __name__ == "__main__":
    file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_1.jpg')
    get_histogram(file)
    