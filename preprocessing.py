from cv2 import cv2
from utils import rgb2hsv
import imutils


def resize(image):
    return imutils.resize(image, height=480)


def preprocessingFrame(image, color):
    y = image.shape[0]

    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsvColor = rgb2hsv(color[2], color[1], color[0])
    h = hsvColor[0] * 255
    s = hsvColor[1] * 255
    v = hsvColor[2] * 255

    hMin = max(h - 20, 0)
    hMax = min(h + 20, 255)
    sMin = max(s - 75, 0)
    sMax = min(s + 75, 255)
    vMin = max(v - 50, 0)
    vMax = min(v + 50, 255)

    if(s < 70):
        hMin = 0
        hMax = 255

    mask = cv2.inRange(hsvImage, (hMin, sMin, vMin), (hMax, sMax, vMax))
    filtered = cv2.bitwise_and(hsvImage, hsvImage, mask=mask)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(y / 75), round(y / 75))))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (round(y/100), round(y/100))), iterations=3)

    return closed
