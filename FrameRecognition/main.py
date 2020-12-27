from collections import defaultdict
import numpy as np
import argparse
import glob
import math
from cv2 import cv2
import imutils
import random
import colorsys


def isEqual(l1,  l2):

    length1 = math.sqrt((l1.x2 - l1.x1)*(l1.x2 - l1.x1) +
                        (l1.y2 - l1.y1)*(l1.y2 - l1.y1))
    length2 = math.sqrt((l2.x2 - l2.x1)*(l2.x2 - l2.x1) +
                        (l2.y2 - l2.y1)*(l2.y2 - l2.y1))

    product = (l1.x2 - l1.x1)*(l2.x2 - l2.x1) + (l1.y2 - l1.y1)*(l2.y2 - l2.y1)

    if (abs(product / (length1 * length2)) < math.cos(math.pi / 60)):
        return False

    # mx1 = (l1.x1 + l1.x2) * 0.5
    # mx2 = (l2.x1 + l2.x2) * 0.5

    # my1 = (l1.y1 + l1.y2) * 0.5
    # my2 = (l2.y1 + l2.y2) * 0.5
    # dist = math.sqrt((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2)

    m1 = (l1.y2 - l1.y1) / (l1.x2 - l1.x1)
    m2 = (l2.y2 - l2.y1) / (l2.x2 - l2.x1)

    q1 = -(m1*l1.x1 + l1.y1)
    q2 = -(m2*l2.x1 + l2.y1)

    dist = abs(q1 - q2) / math.sqrt(1 + abs(m1*m2))

    # if (dist > max(length1, length2) * 0.5):
    if (dist > 100):
        return False

    return True


def hsv2rgb(h,  s,  v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,  s,  v))


class Line():
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


image = cv2.imread("2.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_image = cv2.GaussianBlur(hsv_image, (9, 9), 0)

# mask1 = cv2.inRange(hsv_image, (20, 0, 0), (120, 255, 255)) # Questo trova le api
mask1 = cv2.inRange(hsv_image, (10, 95, 170), (20, 255, 255))
target = cv2.bitwise_and(image, image, mask=mask1)
# cv2.imshow("target.png", target)

image = imutils.resize(target, height=300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
total = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)

canny = cv2.Canny(total, 225, 250)
cv2.imshow("canny", canny)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
# minimum number of pixels making up a line
min_line_length = image.shape[0] * 0.4
# maximum gap in pixels between connectable line segments
max_line_gap = min_line_length*0.5
line_image = np.copy(image) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(canny, rho, theta, threshold,
                        np.array([]), min_line_length, max_line_gap)

linee = []

for line in lines:
    for x1, y1, x2, y2 in line:
        linee.append(Line(x1, y1, x2, y2))

newLines = []
for i in range(0, len(linee)-1):
    newLines.append([linee[i]])
    for j in range(i+1, len(linee)):
        if isEqual(linee[i], linee[j]):
            newLines[i].append(linee[j])

groups = []

for i, lines in enumerate(newLines):  # O(n^34567890)
    groups.append([])
    for line in lines:
        x = False
        for group in groups:
            for groupLine in group:
                if(groupLine == line):
                    x = True
        if not x:
            groups[i].append(line)

groups = [x for x in groups if x != []]

for i, group in enumerate(groups):
    h = (len(groups) - i) / len(groups)
    s = 1
    v = 1
    for line in group:
        cv2.line(line_image, (line.x1, line.y1),

                 (line.x2, line.y2), hsv2rgb(h, s, v), 2)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(image, 0.2, line_image, 1, 0)
cv2.imshow("lines_edges", lines_edges)

cv2.waitKey(0)
