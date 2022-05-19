import matlab.engine
import numpy as np
from PIL import Image
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# split full mask into separate masks for each line
def split_masks(original_mask):
    values = np.unique(original_mask)
    original_mask = np.array(original_mask)
    return_masks = []

    for val in values:
        if val != 0.:
            return_masks.append((original_mask == val).astype(float))

    return return_masks


# order masks from top to bottom to go by line occurrence order
def sort_masks(separated_masks):
    heights = []

    for mask in separated_masks:
        height = 0
        count = 0

        for i in range(len(mask)):
            for e in mask[i][:]:
                if e == 1:
                    height += i
                    count += 1

        heights.append(height/count)

    return [separated_masks for _, separated_masks in sorted(zip(heights, separated_masks))]


# returns full mask for binarized image
def extract_complete_mask(file):
    path = os.path.join(ROOT_DIR, "LineExtraction2")
    eng = matlab.engine.start_matlab()
    eng.cd(path, nargout=0)
    result, Labels, linesMask, colouredMask = eng.run_barakat_from_python(file, nargout=4)

    x = np.array(colouredMask._data).reshape(colouredMask.size[::-1]).T

    return x


# extracts masks from binarized image and returns all masks separately
def extract_masks(file):
    path = os.path.join(ROOT_DIR, "LineExtraction2")
    eng = matlab.engine.start_matlab()
    eng.cd(path, nargout=0)
    result, Labels, linesMask, colouredMask = eng.run_barakat_from_python(file, nargout=4)

    x = np.array(colouredMask._data).reshape(colouredMask.size[::-1]).T

    # returns binary masks of each separate line
    separated_masks = split_masks(x)

    separated_masks = sort_masks(separated_masks)

    return separated_masks


if __name__ == "__main__":
    separated_masks = extract_masks('C:\GitHub_Jeroen\Handwriting_Recognition\data\\task1\image-data\P106-Fg002-R-C01-R01-binarized.jpg')
    i = 1

    for mask in separated_masks:
        mask *= (255 / mask.max())
        img = Image.fromarray(mask.astype(np.uint8))
        # img.show()
        img.save("python_mask_" + str(i) + ".jpg")
        i += 1

    # np.save("test_save.jpg", separated_masks)
