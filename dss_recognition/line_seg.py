import cv2
import numpy as np
import glob, os

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# change working directory
os.chdir(os.getcwd() + "/data/image-data")

# specify angles of rotation
rotations = [0, 1, 2, 3, -1, -2, -3]

for file in glob.glob("*binarized.jpg"):
    original_image = cv2.imread(file, cv2.IMREAD_UNCHANGED) 
    print("FILE:", file)
    for angle in rotations:
        print("\tANGLE:", angle)
        # invert the image and rotate it
        img = original_image.copy()
        img = cv2.bitwise_not(img)
        img = rotate_image(img, angle)

        height, width = img.shape
        
        # get the horizontal projection by summing all values in each row
        horizontal_proj = np.sum(img,1)

        # get the maximum value of all rows to scale the projection image
        max_val = np.max(horizontal_proj)
        proj_img = np.zeros((height, width), np.uint8)

        for row in range(height):
            # draw a line based on how many black pixels each row contains
            cv2.line(proj_img, (0, row), (int(horizontal_proj[row] * width/max_val),row), (255,255,255), 1)
        
        # concatenate the two images to show them simultaneously 
        horizontal = np.concatenate((img, proj_img), axis=1)
        window_name = "Horizontal histogram projection at angle " + str(angle)
        cv2.imshow(window_name, horizontal)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
