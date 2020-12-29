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
    product = (l1.x2 - l1.x1)*(l2.x2 - l2.x1) + (l1.y2 - l1.y1)*(l2.y2 - l2.y1)

    if (abs(product / (l1.getLenght() * l2.getLenght())) < math.cos(math.pi / 60)):
        return False

    a1, b1, c1 = l1.getEquation()
    a2, b2, c2 = l2.getEquation()

    dist = math.inf

    if(a1/b1 == math.nan or abs(a1/b1) > 1):
        dist = abs(c1/a1 - c2/a2) / math.sqrt(1 + abs((b1/a1) * (b2/a2)))
    else:
        dist = abs(c1/b1 - c2/b2) / math.sqrt(1 + abs((a1/b1) * (a2/b2)))

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

    def getM(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def getQ(self):
        return -(self.getM()*self.x1 + self.y1)

    def getLenght(self):
        return math.sqrt((self.x2 - self.x1)*(self.x2 - self.x1) + (self.y2 - self.y1)*(self.y2 - self.y1))

    def getEquation(self):
        a = self.y1 - self.y2
        b = self.x2 - self.x1
        c = (self.x1 - self.x2)*self.y1 + (self.y2 - self.y1)*self.x1
        return (a, b, c)


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

newLines.append([linee[-1]])

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

groups = [x for x in groups if x != []] # cancello le liste vuote

for i, group in enumerate(groups):
    h = (len(groups) - i) / len(groups)
    s = 1
    v = 1
    for line in group:
        cv2.line(line_image, (line.x1, line.y1),
                 (line.x2, line.y2), hsv2rgb(h, s, v), 2)

lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
cv2.imshow("lines_edges", lines_edges)

frameLines = []

for group in groups:
    totalA = 0
    totalB = 0
    totalC = 0

    for i, line in enumerate(group):
        a, b, c = line.getEquation() # ax + by + c = 0

        totalA += a
        totalB += b
        totalC += c

    a = totalA/len(group)
    b = totalB/len(group)
    c = totalC/len(group)

    y1 = 0
    y2 = 0

    if (b != 0):
        x1 = 0
        y1 = -c / b

        x2 = image.shape[1]
        y2 = -(c + a*x2) / b

    if (y1 > 0 and y1 < image.shape[0] and y2 > 0 and y2 < image.shape[0]):
        frameLines.append(Line(int(round(x1)), int(round(y1)),
                               int(round(x2)), int(round(y2))))
    else:
        y1 = 0
        x1 = -c/a

        y2 = image.shape[0]
        x2 = -(c + b*y2) / a

        frameLines.append(Line(int(round(x1)), int(round(y1)),
                               int(round(x2)), int(round(y2))))

grouped_image = np.copy(image) * 0

for line in frameLines:
    cv2.line(grouped_image, (line.x1, line.y1),
             (line.x2, line.y2), (255, 255, 255), 2)

grouped_lines = cv2.addWeighted(image, 0.8, grouped_image, 1, 0)
cv2.imshow("grouped_lines", grouped_lines)

for line in frameLines:
    cv2.line(line_image, (line.x1, line.y1),
             (line.x2, line.y2), (255, 255, 255), 2)

mix = cv2.addWeighted(image, 0.8, line_image, 1, 0)
cv2.imshow("mix", mix)

cv2.waitKey(0)
