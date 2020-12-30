from cv2 import cv2
import imutils


def resize(image):
    return imutils.resize(image, height=300)


def preprocessingFrame(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsvImage, (9, 9), 0)

    mask = cv2.inRange(blurred, (10, 95, 170), (20, 255, 255))
    filtered = cv2.bitwise_and(blurred, blurred, mask=mask)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)

    return closed
