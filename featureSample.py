from mousePoints import MousePoints
import numpy as np
import math
from cv2 import cv2
import matplotlib.pyplot as plt
from utils import getHistogram


def getFeatureSampleAverageColour(image):
    cv2.imshow("Select", image)
    coordinateStore = MousePoints("Select", image)

    points, _ = coordinateStore.getpt(1)
    (x, y) = points[0][0], points[0][1]

    cv2.destroyWindow("Select")

    boxDim = 20

    if (x < boxDim):
        x = boxDim / 2

    if (y < boxDim):
        y = boxDim / 2

    if (x > image.shape[1] - boxDim):
        x = image.shape[1] - boxDim

    if (y > image.shape[0] - boxDim):
        y = image.shape[0] - boxDim

    crop_img = image[round(y-boxDim/2):round(y+boxDim/2),
                     round(x-boxDim/2):round(x+boxDim/2)]

    mask = np.ones([boxDim, boxDim], np.uint8)
    mean = cv2.mean(crop_img, mask=mask)

    return mean


def getFeatureSampleHistogram(image):
    cv2.imshow("Select", image)
    coordinateStore = MousePoints("Select", image)

    points, _ = coordinateStore.getpt(1)
    (x, y) = points[0][0], points[0][1]

    cv2.destroyWindow("Select")

    boxDim = 25

    if (x < boxDim):
        x = boxDim / 2

    if (y < boxDim):
        y = boxDim / 2

    if (x > image.shape[1] - boxDim):
        x = image.shape[1] - boxDim

    if (y > image.shape[0] - boxDim):
        y = image.shape[0] - boxDim

    crop_img = image[round(y-boxDim/2):round(y+boxDim/2),
                     round(x-boxDim/2):round(x+boxDim/2)]

    hist = getHistogram(crop_img)

    return hist
