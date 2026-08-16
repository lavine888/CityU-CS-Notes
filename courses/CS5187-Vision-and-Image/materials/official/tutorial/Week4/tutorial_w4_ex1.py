import cv2


def mosaic_face(img):
    # INSERT YOUR CODE HERE
    pass


if __name__ == "__main__":
    img = cv2.imread("tst1.jpg")
    result = mosaic_face(img)
    cv2.imshow('original', img)
    cv2.imshow('mosaic', result)
    cv2.waitKey(0)
