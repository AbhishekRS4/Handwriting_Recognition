import matlab.engine
import numpy as np
from PIL import Image

def split_masks(original_mask):
    print("hey")
    values = np.unique(original_mask)
    original_mask = np.array(original_mask)
    print(values)
    return_masks = []
    
    for val in values:
        if val != 0.:
            print("yo")
            return_masks.append((original_mask == val).astype(float))
    
    return return_masks

eng = matlab.engine.start_matlab()
result, Labels, linesMask, colouredMask = eng.run_barakat_from_python(nargout=4)

x = np.array(colouredMask._data).reshape(colouredMask.size[::-1]).T

# returns binary masks of each separate line
separated_masks = split_masks(x)

# img = Image.fromarray(separated_masks[0], mode='1')
i = 1
for mask in separated_masks:
    mask *= (255/mask.max())
    img = Image.fromarray(mask.astype(np.uint8))
    # img.show()
    img.save("python_mask_" + str(i) + ".jpg")
    i += 1

np.save("test_save", separated_masks)
