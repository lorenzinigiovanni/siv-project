import colorsys
from cv2 import cv2
import numpy as np


def hsv2rgb(h,  s,  v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,  s,  v))


def getColorMean(image):
    mask = np.ones([image.shape[0], image.shape[1]], np.uint8)
    mean = cv2.mean(image, mask=mask)
    return mean


def getHistogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
