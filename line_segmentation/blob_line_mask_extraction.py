import LineExtraction2.run_matlab_code as rmc
import dataset_task1 as ds
import numpy as np
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TODO:
#   -   allow for reading of files DONE
#   -   order masks from top to bottom DONE
#   -   extract lines from images
#   -


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
        img.save(f"{get_subdirectory('masks')}/image_" + str(index) + "_python_mask_" + str(i) + ".jpg")
        i += 1
    # array of binary masks
    # np.save("results/image_" + str(index) + "full.jpg", masks)


# get masks from the input data and save them
def get_masks():
    file_names = ds.read_binarized_images()

    i = 1

    for file in file_names:
        print(file)
        separated_masks = rmc.extract_masks(file)
        save_masks(separated_masks, i)
        i += 1


if __name__ == "__main__":
    get_masks()
    # separated_masks = rmc.extract_masks(
    #     'C:\GitHub_Jeroen\Handwriting_Recognition\data\\task1\image-data\P106-Fg002-R-C01-R01-binarized.jpg')
    # i = 1
    #
    # for mask in separated_masks:
    #     mask *= (255 / mask.max())
    #     img = Image.fromarray(mask.astype(np.uint8))
    #     # img.show()
    #     img.save("python_mask_" + str(i) + ".jpg")
    #     i += 1
