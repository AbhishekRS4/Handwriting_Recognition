import cv2
import numpy as np
import os
import copy

def get_subdirectory(sd):
    dir = os.path.join(os.getcwd(), f'{sd}')
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def line_segmentation(horizontal_proj, average_filter_width, threshold, alpha):
    # applying filter
    horizontal_proj = moving_average(horizontal_proj, average_filter_width)

    # copying projection, orderering it, and getting the indices of ordered projection
    copied_proj = copy.copy(horizontal_proj)
    ordered_proj = np.sort(copied_proj)[::-1]
    ordered_proj_indices = np.argsort(copied_proj)[::-1]

    idx = 0
    set_A = set()
    set_B = []
    peak_heights = []

    while ordered_proj[idx] > (alpha * np.max(ordered_proj)):
        if ordered_proj_indices[idx] not in set_A:
            adaptive_threshold = threshold * ordered_proj[idx] # base threshold on height of peak.

            # find left coordinate of width of peak.
            left = 1
            while horizontal_proj[ordered_proj_indices[idx] - left] >= adaptive_threshold:
                left += 1
            x1 = ordered_proj_indices[idx] - left + 1

            # find right coordinate of width of peak.
            right = 1
            while horizontal_proj[ordered_proj_indices[idx] + right] <= adaptive_threshold:
                right += 1
            x2 = ordered_proj_indices[idx] + right - 1

            # all indices that fall within peak range.
            set_R = set(range(x1, x2+1))

            # if the found coordinates are not in A yet, add them to set_B.
            if len(set_R.intersection(set_A)) == 0:
                peak_heights.append(ordered_proj[idx])
                set_B.append([x1, x2]) # add interval R={x1,x2} to the set of intervals.

            set_A = set_A.union(set_R)
        idx += 1
    # Sort the found ranges from top peak to bottom peak.
    set_B.sort()
    set_S = [] # set with separation points.

    # At this point, set_B contains all the indices that fall within peaks.
    for i in range(0, len(set_B) - 1):
        range_between_interval = list(range(set_B[i][1], set_B[i+1][0] + 1)) # get index values of area between intervals.
        height_values = [horizontal_proj[index] for index in range_between_interval] # get the lowest height value of those indices.
        set_S.append(range_between_interval[height_values.index(np.min(height_values))]) # get the index of this lowest height value, line separator found.
    
    # Set_S contains all separating lines.
    return horizontal_proj, set_S, np.mean(peak_heights)

def get_best_image(img_information):
    peak_heights = [information[2] for information in img_information] # get average peak heights of images rotated at different angles.
    index_of_best_image = peak_heights.index(np.max(peak_heights)) # the image with the highest average peak height has the most pronounced peaks -> best histogram/image

    return img_information[index_of_best_image] # return all useful attributes

# def crop_image(img):
#     for index, row in enumerate(img):
#         if cv2.countNonZero(row) > 100:
#             print(row)
#             top_border = index
#             break

#     for index, row in enumerate(img[::-1]):
#         if cv2.countNonZero(row) > 100:
#             bottom_border = index
#             break
    
#     for index, column in enumerate(img.T):
#         if cv2.countNonZero(column) > 100:
#             right_border = index
#             break

#     for index, column in enumerate(img.T[::-1]):
#         if cv2.countNonZero(column) > 100:
#             left_border = index
#             break
#     left_border = 0
#     right_border = 0
#     print(top_border, bottom_border, left_border, right_border)
#     print(img.shape)

    # img = img[top_border:img.shape[0]-bottom_border, left_border:img.shape[1]-right_border]
    # return img

def extract_line_images(img, separating_lines, index, write_images = False):
    line_images = []
    # we get as many images as separating lines + 1
    for idx in range(len(separating_lines) + 1):
        if idx == 0:
            line_images.append(img[0:separating_lines[idx], 0:img.shape[1]])
        elif idx == len(separating_lines):
            line_images.append(img[separating_lines[idx-1]:img.shape[0], 0:img.shape[1]])
        else:
            line_images.append(img[separating_lines[idx-1]:separating_lines[idx], 0:img.shape[1]])

    # crop each image
    for idx, line_image in enumerate(line_images):
        line_image = cv2.bitwise_not(line_image).copy()
        if write_images:
            print("writing as: ", f"{get_subdirectory('crops')}/image_" + str(index) + "_crop_" + str(idx) + ".jpg")
            cv2.imwrite(f"{get_subdirectory('crops')}/image_" + str(index) + "_crop_" + str(idx) + ".jpg", line_image)
    return line_images


def extract_lines(file):
    # specify angles of rotation.
    rotations = [0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5]
    index = 0
    original_image = cv2.imread(file, cv2.IMREAD_UNCHANGED) 
    img_information = []
    for angle in rotations:
        # invert the image and rotate it.
        img = original_image.copy()
        img = cv2.bitwise_not(img).copy()

        img = rotate_image(img, angle)
        height, width = img.shape

        # get the horizontal projection by summing all values in each row.
        horizontal_proj = np.sum(img, axis = 1)

        horizontal_proj, separating_lines, mean_peak_height = line_segmentation(horizontal_proj, 50, 0.80, 0.05)
        img_information.append([img, horizontal_proj, mean_peak_height, angle, separating_lines])

    img, horizontal_proj, mean_peak_height, angle, separating_lines = get_best_image(img_information)
    # get the maximum value of all rows to scale the projection image.
    max_val = np.max(horizontal_proj)
    proj_img = np.zeros((height, width), np.uint8)

    line_images = extract_line_images(img, separating_lines, index)

    return line_images
    # # draw a line based on how many black pixels each row contains.
    # for row in range(height):
    #     cv2.line(proj_img, (0, row), (int(horizontal_proj[row] * width/max_val),row), (255,255,255), 1)
    
    # # draw a line for each found separating line.
    # for separating_line in separating_lines:
    #     cv2.line(img, (0, separating_line), (width, separating_line), (255,0,0), 3)

    # # concatenate the two images to show them simultaneously.
    # horizontal = np.concatenate((img, proj_img), axis=1)
    # window_name = "Horizontal histogram projection at angle " + str(angle)
    # cv2.imshow(window_name, horizontal)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    pass