import numpy as np
import argparse
import glob
import math
from cv2 import cv2
import imutils
import random

from line import Line
from borders import houghLines, makeGroups, mathLines
from preprocessing import preprocessingFrame, resize
from perspective import fourPointTransform, orderPoints, warpedImage


source = cv2.imread("Images/02.png")
resized = resize(source)

preprocessedFrame = preprocessingFrame(resized)
houghLines = houghLines(preprocessedFrame)
groups = makeGroups(houghLines)
lines = mathLines(groups, preprocessedFrame.shape[1], preprocessedFrame.shape[0])
warped = warpedImage(lines, resized)

cv2.imshow("warped", warped)

cv2.waitKey(0)

# print("Select frame:")
# print("Select bee:")
# print("Select open cell:")
# print("Select closed cell:")
