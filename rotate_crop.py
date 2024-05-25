#from image import Image
#from functions import *
from statistics import median
from math import inf
import cv2
from image import Image

def threshold(img, s):
    """
    Apply a binary threshold to the image.
    :param img: an image as defined in OpenCV
    :param s: threshold value
    :return: thresholded image
    """
    _, I = cv2.threshold(img, s, 255, cv2.THRESH_BINARY)
    return I

def rotate(image, angle, center=None, scale=1.0):
    """
    Rotates an image by the given angle.
    :param image: The image to be rotated.
    :param angle: The angle by which to rotate the image.
    :param center: The center of rotation. If None, the center of the image is used.
    :param scale: The scale factor.
    :return: The rotated image.
    """
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def rotate_neck_picture(image):
    """
    Rotating the picture so that the neck of the guitar is horizontal. We use Hough transform to detect lines
    and calculating the slopes of all lines, we rotate it according to the median slope.
    Hopefully, most lines detected will be strings or neck lines so the median slope is the slope of the neck
    An image with lots of noise and different lines will result in poor results.
    :param image: an Image object
    :return rotated_neck_picture: an Image object rotated according to the angle of the median slope detected in param image
    """
    image_to_rotate = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    lines = image.lines_hough_transform(edges, 50, 50)

    if lines is None or len(lines) == 0:
        print("No lines detected.")
        return image  # Return the original image if no lines are detected

    slopes = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append(abs((y2 - y1) / (x2 - x1)))

    if not slopes:
        print("No slopes detected.")
        return image  # Return the original image if no slopes are detected

    median_slope = median(slopes)
    angle = median_slope * 45

    return Image(img=rotate(image_to_rotate, -angle))


def crop_neck_picture(image):
    """
    Cropping the picture so we only work on the region of interest (i.e. the neck)
    We're looking for a very dense region where we detect horizontal line
    Currently, we identify it by looking at parts where there are more than two lines at the same y (height)
    :param image: an Image object of the neck (rotated horizontally if necessary)
    :return cropped_neck_picture: an Image object cropped around the neck
    """
    image_to_crop = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    lines = image.lines_hough_transform(edges, 50, 50)  # TODO: Calibrate params automatically
    if lines is None or len(lines) == 0:
        print("No lines detected for cropping.")
        return image 
    y = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            y.append(y1)
            y.append(y2)

    y_sort = list(sorted(y))
    y_differences = [0]

    first_y = None
    last_y = None

    for i in range(len(y_sort) - 1):
        y_differences.append(y_sort[i + 1] - y_sort[i])
    for i in range(len(y_differences) - 1):
        if y_differences[i] == 0:
            last_y = y_sort[i]
            if i > 3 and first_y is None:
                first_y = y_sort[i]

    if first_y is None or last_y is None:
        print("Failed to determine crop boundaries.")
        return None

    if first_y - 10 < 0 or last_y + 10 > image_to_crop.shape[0]:
        print("Crop boundaries are out of image bounds.")
        return None

    cropped_img = image_to_crop[first_y - 10:last_y + 10]
    if cropped_img.size == 0:
        print("Cropped image is empty.")
        return None

    return Image(img=cropped_img)


def resize_image(img):
    """
    Recursive function to resize image if definition is too elevated
    :param img: an image as defined in OpenCV
    :return: an image as defined in OpenCV
    """
    height = len(img)
    width = len(img[0])
    if height >= 1080 or width >= 1920:
        resized_image = cv2.resize(img, (int(width * 0.8), int(height * 0.8)))
        return resize_image(resized_image)
    else:
        return img


if __name__ == "__main__":
    print("Run rotate_crop_tests.py to have a look at results!")