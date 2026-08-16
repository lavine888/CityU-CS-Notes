import cv2


def cartoonize_image(img):
    # INSERT YOUR CODE HERE
    pass


if __name__ == "__main__":
    img = cv2.imread("tst3.jpg")
    result = cartoonize_image(img)
    cv2.imshow('original', img)
    cv2.imshow('cartoonized', result)
    cv2.waitKey(0)
