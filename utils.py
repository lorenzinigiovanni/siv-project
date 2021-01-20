import colorsys
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def hsv2rgb(h,  s,  v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,  s,  v))


def rgb2hsv(r, g, b):
    r = float(r) / 255
    g = float(g) / 255
    b = float(b) / 255
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v


def getColorMean(image):
    mask = np.ones([image.shape[0], image.shape[1]], np.uint8)
    mean = cv2.mean(image, mask=mask)
    return mean


def getHistogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([image], [i], None, [10], [0, 256])
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 10])

    # plt.show()

    return hist
