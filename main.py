import numpy as np
import argparse
import glob
import math
from cv2 import cv2
import imutils
import random
import matplotlib.pyplot as plt

from line import Line
from borders import houghLines, makeGroups, mathLines
from preprocessing import preprocessingFrame, resize
from perspective import fourPointTransform, orderPoints, warpedImage
from featureSample import getFeatureSampleAverageColour, getFeatureSampleHistogram
from utils import getColorMean, getHistogram
from recognition import recognitionByColor, recognitionByHistogram


source = cv2.imread("Images/02.png")
resized = resize(source)

preprocessedFrame = preprocessingFrame(resized)
houghLines = houghLines(preprocessedFrame)
groups = makeGroups(houghLines)
lines = mathLines(
    groups, preprocessedFrame.shape[1], preprocessedFrame.shape[0])
warped = warpedImage(lines, resized)

print("Select bee")
beeColour = getFeatureSampleAverageColour(warped)
beeHistogram = getFeatureSampleHistogram(warped)

print("Select open cell")
openCellColour = getFeatureSampleAverageColour(warped)
openCellHistogram = getFeatureSampleHistogram(warped)

print("Select closed cell")
closeCellColour = getFeatureSampleAverageColour(warped)
closedCellHistogram = getFeatureSampleHistogram(warped)

sixths = []

for i in range(0, 6):
    x = warped.shape[1]
    y = warped.shape[0]

    xStart = (i % 3) * (x / 3)
    yStart = (i % 2) * (y / 2)
    xEnd = (1 + i % 3) * (x / 3)
    yEnd = (1 + i % 2) * (y / 2)

    sixths.append(warped[round(yStart):round(yEnd), round(xStart):round(xEnd)])

colorDiscrimination = recognitionByColor(
    sixths, beeColour, openCellColour, closeCellColour)
print(colorDiscrimination)

histogramDiscrimination = recognitionByHistogram(
    sixths, beeHistogram, openCellHistogram, closedCellHistogram)
print(histogramDiscrimination)

cv2.waitKey(0)
