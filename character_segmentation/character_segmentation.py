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
def split_characters(image, splits):
    characters = []

    min = 0
    print(len(image.T))
    for split in splits:
        characters.append(image[:, min:split])
        min = split

        if split == splits[-1]:
            characters.append(image[:, min:len(image.T)])

    return characters



# this function creates a histogram from the input image and determines the splits based on the binarized histogram
def get_histogram(file):
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

    ax[2].set_title(' binarized ')
    ax[2].stairs(binarized, fill=False, color="orange")
    binary_valleys, _ = sp.find_peaks(np.multiply(binarized, -1))
    ax[2].vlines(x=binary_valleys, ymin=0, ymax=1, colors="red")
    ax[2].set_xlim([0, len(binarized)])

    plt.show()

    characters = split_characters(image, binary_valleys)

    for character in characters:
        cv2.imshow("char", character)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_8.jpg')
    file = os.path.join(ROOT_DIR, 'line_segmentation\crops\image_1_crop_6.jpg')
    get_histogram(file)
