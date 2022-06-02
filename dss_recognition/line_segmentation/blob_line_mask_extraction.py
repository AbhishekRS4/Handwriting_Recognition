from LineExtraction2 import run_matlab_code as rmc
import dataloader_task1 as ds
import numpy as np
import os
import cv2
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_subdirectory(sd):
    dir = os.path.join(os.getcwd(), f'{sd}')
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


# save all masks to the masks/ folder
def save_masks(masks, index):
    i = 1

    for mask in masks:
        mask *= (255 / mask.max())
        img = Image.fromarray(mask.astype(np.uint8))
        # img.show()
        img.save(f"{get_subdirectory('masks')}\image_" + str(index) + "_python_mask_" + str(i) + ".jpg")
        i += 1

    # array of binary masks
    np.save("masks/image_" + str(index) + "full", masks)


# white-out surrounding image from mask
def apply_mask(image, mask):
    for i in range(len(mask)):
        for j in range(len(mask[i][:])):
            if mask[i][j] == 0.:
                image[i][j] = (255,255,255)

    return image


# crop mask and image to outline of mask to reduce image size
def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol

    coords = np.ix_(mask.any(1), mask.any(0))
    return img[coords], mask[coords]


# crop images to masks
def extract_lines(file, masks, idx=0, write_crops=False):
    image = cv2.imread(file)
    lines = []

    # Remove background using bitwise-and operation
    i = 1
    for mask in masks:
        crop_im, crop_mask = get_segment_crop(image, mask=mask)
        result = apply_mask(crop_im, crop_mask)
        lines.append(result)

        if write_crops:
            # all crops are written to line_segmentation/crops/image_filenr_crop_cropnr.jpg
            print("writing as: ", f"{get_subdirectory('crops')}\image_" + str(idx) + "_crop_" + str(i) + ".jpg")
            cv2.imwrite(f"{get_subdirectory('crops')}\image_" + str(idx) + "_crop_" + str(i) + ".jpg", result)

        i += 1

    return lines


# get masks from the input data and save them
def get_cropped_images(dir=None):
    file_names = ds.read_binarized_images(dir)
    i = 1

    for file in file_names:
        print(file)
        separated_masks = rmc.extract_masks(file)
        _ = extract_lines(file, separated_masks, i, write_crops=True)
        # save_masks(separated_masks, i)
        i += 1


if __name__ == "__main__":
    # -- to run regular program
    get_cropped_images(dir="/home/abhishek/Desktop/RUG/hw_recognition/dss_test_images")

    # -- to run for single image
    # file = os.path.join(ROOT_DIR, 'image-data\\P106-Fg002-R-C01-R01-binarized.jpg')
    # separated_masks = rmc.extract_masks(file)
    # extract_lines(file, separated_masks, 0)

    # -- to get outlines on single image
    # file = os.path.join(ROOT_DIR, 'image-data\\P106-Fg002-R-C01-R01-binarized.jpg')
    # mask = rmc.extract_complete_mask(file)
    # np.save("masks/contour_test_mask", mask)

    # mask = np.load("masks/contour_test_mask.npy")
    # image = cv2.imread(file)
    #
    # out1 = skimage.color.label2rgb(mask, image, kind='overlay')
    #
    # result = cv2.resize(out1, (960, 540))  # Resize image
    # cv2.imshow("output", result)
    #
    # cv2.waitKey(0)
    # cv2.imwrite(f"{get_subdirectory('masks')}/complete_mask.jpg", result*255)

    # -- to test cropping
    # file = os.path.join(ROOT_DIR, 'image-data\\P106-Fg002-R-C01-R01-binarized.jpg')
    # # separated_masks = rmc.extract_masks(file)
    #
    # # np.save("masks/cropping_test_masks", separated_masks)
    # separated_masks = np.load("masks/cropping_test_masks.npy")
    #
    # image = cv2.imread(file)
    #
    # for mask in separated_masks:
    #     crop_im, crop_mask = get_segment_crop(image, mask=mask)
    #
    #     result = apply_mask(crop_im, crop_mask)
    #
    #     cv2.imshow("result", result)
    #
    #     cv2.waitKey(0)
