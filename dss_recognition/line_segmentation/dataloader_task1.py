import regex as re
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# this function read the binarized images from the data/task1/image-data folder
def read_binarized_images(dir=None):
    if not dir:
        path = os.path.join(ROOT_DIR, "image-data")
    else:
        path = dir

    images = []

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f) and "binarized" in f:
            images.append(f)

    return images


# this function reads the cropped lines from the line_segmentation/crops folder
def read_cropped_lines():
    images = []
    lines = []
    numbers = []
    line_numbers = []

    current_image = None
    path = os.path.join(ROOT_DIR, "line_segmentation\\crops")

    for filename in os.listdir(path):
        num = int(re.search(r'\d+', filename).group(0))

        if not current_image:
            current_image = num
            numbers.append(num)
        elif num != current_image:
            ordered_lines = [x for _, x in sorted(zip(line_numbers, lines))]

            current_image = num
            numbers.append(num)
            images.append(ordered_lines)

            lines = []
            line_numbers = []

        f = os.path.join(path, filename)

        line_numbers.append(int(re.search(r'\d+', f[::-1]).group()[::-1]))
        lines.append(f)

    images.append(lines)
    ordered_images = [x for _, x in sorted(zip(numbers, images))]

    return ordered_images


if __name__ == "__main__":
    # image_files = read_binarized_images()
    # print(image_files)
    read_cropped_lines()
