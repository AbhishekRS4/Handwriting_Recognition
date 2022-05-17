import os

PATH = os.path.join(os.path.dirname(os.getcwd()), "data\\task1\\image-data")


def read_binarized_images():
    images = []

    for filename in os.listdir(PATH):
        f = os.path.join(PATH, filename)
        # checking if it is a file
        if os.path.isfile(f) and "binarized" in f:
            images.append(f)

    return images


if __name__ == "__main__":
    image_files = read_binarized_images()
    print(image_files)
