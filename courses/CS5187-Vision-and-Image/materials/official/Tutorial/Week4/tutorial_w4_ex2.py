import cv2
import numpy as np


def sharpening(img):
    # INSERT YOUR CODE HERE
    pass


def motion_blur(img):
    # INSERT YOUR CODE HERE
    pass


if __name__ == "__main__":
    img = cv2.imread("tst2.jpg")
    result1 = sharpening(img)
    result2 = motion_blur(img)

    cv2.imshow('original', img)
    cv2.imshow('sharpening', result1)
    cv2.imshow('motion blur', result2)
    cv2.waitKey(0)
