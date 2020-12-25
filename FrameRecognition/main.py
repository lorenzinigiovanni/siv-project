import numpy as np
import argparse
import glob
import math  
from cv2 import cv2
import imutils

def isEqual(l1,  l2):
    l1 = l1[0]
    l2 = l2[0]
    length1 = math.sqrt((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]))
    length2 = math.sqrt((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]))

    product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1])

    if (abs(product / (length1 * length2)) < math.cos(math.pi / 30)):
        return False

    mx1 = (l1[0] + l1[2]) * 0.5
    mx2 = (l2[0] + l2[2]) * 0.5

    my1 = (l1[1] + l1[3]) * 0.5
    my2 = (l2[1] + l2[3]) * 0.5
    dist = math.sqrt((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2))

    if (dist > max(length1, length2) * 0.5):
        return False

    return True

image = cv2.imread("2.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_image = cv2.GaussianBlur(hsv_image, (9, 9), 0)

# mask1 = cv2.inRange(hsv_image, (20, 0, 0), (120, 255, 255)) # Questo trova le api
mask1 = cv2.inRange(hsv_image, (10, 95, 170), (20, 255, 255))
target = cv2.bitwise_and(image,image, mask=mask1)
# cv2.imshow("target.png", target)

image = imutils.resize(target, height = 300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
total = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=3)

canny = cv2.Canny(total, 225, 250)
cv2.imshow("canny", canny)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = image.shape[0] * 0.4 # minimum number of pixels making up a line
max_line_gap = min_line_length*0.5  # maximum gap in pixels between connectable line segments
line_image = np.copy(image) * 0  # creating a blank to draw lines on

cv2.imshow("Canny", canny)

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
    	cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),1)
print(len(lines))

newLines=[]
for i in range (0, len(lines)-1):
    newLines.append([lines[i]])
    for j in range (i+1, len(lines)):
        if isEqual(lines[i], lines[j]):
            newLines[i].append(lines[j])

        print(isEqual(lines[i], lines[j]))


# Draw the lines on the  image
lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
cv2.imshow("lines_edges", lines_edges)

cv2.waitKey(0)
