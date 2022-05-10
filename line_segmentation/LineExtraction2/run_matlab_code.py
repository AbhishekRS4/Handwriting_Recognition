import matlab.engine
import numpy as np
from PIL import Image


def split_masks(original_mask):
    values = np.unique(original_mask)
    original_mask = np.array(original_mask)
    return_masks = []

    for val in values:
        if val != 0.:
            return_masks.append((original_mask == val).astype(float))

    return return_masks


def extract_masks():
    eng = matlab.engine.start_matlab()
    result, Labels, linesMask, colouredMask = eng.run_barakat_from_python(nargout=4)

    x = np.array(colouredMask._data).reshape(colouredMask.size[::-1]).T

    # returns binary masks of each separate line
    separated_masks = split_masks(x)

    return separated_masks


if __name__ == "__main__":
    separated_masks = extract_masks()
    i = 1

    for mask in separated_masks:
        mask *= (255 / mask.max())
        img = Image.fromarray(mask.astype(np.uint8))
        # img.show()
        img.save("python_mask_" + str(i) + ".jpg")
        i += 1

    np.save("test_save", separated_masks)
