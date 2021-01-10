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
from utils import hsv2rgb, rgb2hsv, getColorMean, getHistogram
from recognition import recognitionByColor, recognitionByHistogram
from mousePoints import MousePoints


source = cv2.imread("Images/02.png")
frameAuto = False

resized = resize(source)

if(frameAuto):
    print("Select frame")
    frameColor = getFeatureSampleAverageColour(resized)

    preprocessedFrame = preprocessingFrame(resized, frameColor)
    houghLines = houghLines(preprocessedFrame)
    groups = makeGroups(houghLines)

    line_image = np.copy(resized) * 0

    for i, group in enumerate(groups):
        h = (len(groups) - i) / len(groups)
        s = 1
        v = 1
    for line in group:
        cv2.line(line_image, (line.x1, line.y1),
                 (line.x2, line.y2), hsv2rgb(h, s, v), 2)

    lines_edges = cv2.addWeighted(resized, 0.8, line_image, 1, 0)
    cv2.imshow("lines_edges", lines_edges)

    lines = mathLines(
        groups, preprocessedFrame.shape[1], preprocessedFrame.shape[0])
    warped = warpedImage(lines, resized)
else:
    corners = []
    coordinateStore = MousePoints("Select", resized)

    print("Select inner corners of the frame")
    corners, _ = coordinateStore.getpt(4)

    pts = np.array(corners, dtype="float32")
    warped = fourPointTransform(resized, pts)

print("Select bee")
# beeColour = getFeatureSampleAverageColour(warped)
beeHistogram = getFeatureSampleHistogram(warped)

print("Select open cell")
# openCellColour = getFeatureSampleAverageColour(warped)
openCellHistogram = getFeatureSampleHistogram(warped)

print("Select closed cell")
# closeCellColour = getFeatureSampleAverageColour(warped)
closedCellHistogram = getFeatureSampleHistogram(warped)

sixths = []

for yy in range(0, 2):  
    for xx in range(0, 3):
        x = warped.shape[1]
        y = warped.shape[0]

        xStart = xx * (x / 3)
        yStart = yy * (y / 2)
        xEnd = (1 + xx) * (x / 3)
        yEnd = (1 + yy) * (y / 2)

        sixths.append(warped[round(yStart):round(yEnd), round(xStart):round(xEnd)])

# colorDiscrimination = recognitionByColor(
#     sixths, beeColour, openCellColour, closeCellColour)
# print(colorDiscrimination)

histogramDiscrimination = recognitionByHistogram(
    sixths, beeHistogram, openCellHistogram, closedCellHistogram)
print(histogramDiscrimination)

cv2.waitKey(0)
