from mousePoints import MousePoints
import numpy as np
import math
from cv2 import cv2
import matplotlib.pyplot as plt
from utils import getHistogram


def getFeatureSampleAverageColour(image):
    y = image.shape[0]
    boxDim = round(y / 15)
    
    if (boxDim % 2 == 1):
        boxDim += 1

    cv2.imshow("Select", image)
    coordinateStore = MousePoints("Select", image, boxDim)

    points, boxDim = coordinateStore.getpt(1)
    (x, y) = points[0][0], points[0][1]

    cv2.destroyWindow("Select")

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
    y = image.shape[0]
    boxDim = round(y / 10)

    if (boxDim % 2 == 1):
        boxDim += 1

    cv2.imshow("Select", image)
    coordinateStore = MousePoints("Select", image, boxDim)

    points, boxDim = coordinateStore.getpt(1)
    (x, y) = points[0][0], points[0][1]

    cv2.destroyWindow("Select")

    if (x, y) == (-1, -1):
        return None

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
