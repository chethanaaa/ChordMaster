import os
import time
from matplotlib import pyplot as plt
#from image import Image
#from rotate_crop import rotate_neck_picture, crop_neck_picture
from grid_detection import string_detection, fret_detection
import cv2
import numpy as np


class Image:
    """
    This Image object was made at the beginning to simplify the code by packaging every common
    image processing treatment in a single object
    """
    def __init__(self, path=None, img=None):
        if img is None and path is not None:
            self.image = cv2.imread(path)
            if self.image is None:
                print(f"Failed to load image from path: {path}")
        elif img is not None and path is None:
            self.image = img
        else:
            self.image = None
            print("Incorrect image parameter or failed to load image")

        if self.image is not None:
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = None

    def __str__(self):
        return str(self.image)

    def set_image(self, img):
        self.image = img

    def set_gray(self, grayscale):
        self.gray = grayscale

    def print_cv2(self, is_gray=True):
        if is_gray:
            cv2.imshow('image', self.gray)
        else:
            cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print_plt(self, is_gray=True):
        if is_gray:
            plt.imshow(self.gray, cmap='gray')
        else:
            plt.imshow(self.image)
        plt.show()

    def edges_canny(self, min_val=100, max_val=200, aperture=3):
        return cv2.Canny(self.gray, min_val, max_val, aperture)

    def edges_laplacian(self, ksize=3):
        return cv2.Laplacian(self.gray, cv2.CV_8U, ksize)

    def edges_sobelx(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 1, 0, ksize)

    def edges_sobely(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 0, 1, ksize)

    def lines_hough_transform(self, edges, min_line_length, max_line_gap, threshold=15):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
        return lines if lines is not None else []
    
    
def string_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_string = string_detection(cropped_image)[1]
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_string.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()


def fret_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_fret = fret_detection(cropped_image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_fret.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()


def grid_detection_tests():
    i = 1
    plt.figure(1)
    for filename in os.listdir('./pictures/'):
        print("File found: " + filename + " - Processing...")
        start_time = time.time()
        chord_image = Image(path='./pictures/' + filename)
        rotated_image = rotate_neck_picture(chord_image)
        cropped_image = crop_neck_picture(rotated_image)
        neck_strings = string_detection(cropped_image)[0]
        neck_fret = fret_detection(cropped_image)
        for string, pts in neck_strings.separating_lines.items():
            cv2.line(neck_fret.image, pts[0], pts[1], (127, 0, 255), 2)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_fret.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()


if __name__ == "__main__":
    print("What would you like to detect? \n\t1 - Strings \n\t2 - Frets \n\t3 - Strings and frets")
    choice = input("[1/2/3] > ")
    if choice == "1":
        print("Detecting strings...")
        string_detection_tests()
    elif choice == "2":
        print("Detecting frets...")
        fret_detection_tests()
    elif choice == "3":
        print("Detecting whole grid...")
        grid_detection_tests()
    else:
        print("Command not defined - Aborted.")