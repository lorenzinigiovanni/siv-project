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
from results import showResults
from kmeans import sklearnKMeans, cv2KMeans


sizeX = 3
sizeY = 2
frameAuto = True

source = cv2.imread("dataset/02.png")

resized = resize(source)
warped = resized

if(frameAuto):
    print("Select frame")
    frameColor = getFeatureSampleAverageColour(resized)

    linesDistance = round(resized.shape[0] / 12)

    preprocessedFrame = preprocessingFrame(resized, frameColor)
    houghLines = houghLines(preprocessedFrame)
    groups = makeGroups(houghLines, linesDistance)

    line_image = np.copy(resized) * 0

    for i, group in enumerate(groups):
        h = (len(groups) - i) / len(groups)
        s = 1
        v = 1
        for line in group:
            cv2.line(line_image, (line.x1, line.y1),
                     (line.x2, line.y2), hsv2rgb(h, s, v), 2)

    lines = mathLines(
        groups, preprocessedFrame.shape[1], preprocessedFrame.shape[0])

    for i, line in enumerate(lines):
        cv2.line(line_image, (line.x1, line.y1),
                 (line.x2, line.y2), (255, 255, 255), 2)

    lines_edges = cv2.addWeighted(resized, 0.8, line_image, 1, 0)
    cv2.imshow("lines_edges", lines_edges)

    warped = warpedImage(lines, resized)
else:
    corners = []
    coordinateStore = MousePoints("Select inner corners of the frame", resized)

    print("Select inner corners of the frame")
    corners, _ = coordinateStore.getpt(4)
    cv2.destroyWindow("Select inner corners of the frame")

    pts = np.array(corners, dtype="float32")
    warped = fourPointTransform(resized, pts)

print("You can now select as requested")
print("Select with left click")
print("Skip with right click")
print("Change box dimension with scrolwheel")

print("Select bee")
beeColour = getFeatureSampleAverageColour(warped)
beeHistogram = getFeatureSampleHistogram(warped)

print("Select open cell")
openCellColour = getFeatureSampleAverageColour(warped)
openCellHistogram = getFeatureSampleHistogram(warped)

print("Select close cell")
closeCellColour = getFeatureSampleAverageColour(warped)
closeCellHistogram = getFeatureSampleHistogram(warped)

sixths = []

for yy in range(0, sizeY):
    for xx in range(0, sizeX):
        x = warped.shape[1]
        y = warped.shape[0]

        xStart = xx * (x / sizeX)
        yStart = yy * (y / sizeY)
        xEnd = (1 + xx) * (x / sizeX)
        yEnd = (1 + yy) * (y / sizeY)

        sixths.append(warped[round(yStart):round(
            yEnd), round(xStart):round(xEnd)])

colorDiscrimination = recognitionByColor(
    sixths, beeColour, openCellColour, closeCellColour)

histogramDiscrimination = recognitionByHistogram(
    sixths, beeHistogram, openCellHistogram, closeCellHistogram)

showResults(colorDiscrimination,
            "Color Discrimination", warped, sizeX, sizeY)
showResults(histogramDiscrimination,
            "Histogram Discrimination", warped, sizeX, sizeY)

skKmeansImage = sklearnKMeans(warped)
cv2KmeansImage = cv2KMeans(warped)

cv2.imshow("skKmeansImage", skKmeansImage)
cv2.imshow("cv2KmeansImage", cv2KmeansImage)

cv2.waitKey(0)
