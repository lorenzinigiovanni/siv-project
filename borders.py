from cv2 import cv2
import numpy as np

from line import Line


def houghLines(image):
    y = image.shape[0]
    
    canny = cv2.Canny(image, 225, 250)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 50
    # minimum number of pixels making up a line
    min_line_length = y * 0.4
    # maximum gap in pixels between connectable line segments
    max_line_gap = min_line_length * 0.5

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(canny, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)

    linee = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            linee.append(Line(x1, y1, x2, y2))

    return linee


def makeGroups(lines, linesDistance):
    newLines = []
    for i in range(0, len(lines)-1):
        newLines.append([lines[i]])
        for j in range(i+1, len(lines)):
            if lines[i].isEqual(lines[j], linesDistance):
                newLines[i].append(lines[j])

    newLines.append([lines[-1]])

    groups = []
    while len(newLines) > 0:
        first, *rest = newLines
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        groups.append(first)
        newLines = rest

    groups = [x for x in groups if x != []]
    return groups


def mathLines(groups, sizeX, sizeY):
    frameLines = []

    for group in groups:
        totalA = 0
        totalB = 0
        totalC = 0

        for line in group:
            a, b, c = line.getEquation()

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

            x2 = sizeX
            y2 = -(c + a*x2) / b

        if (y1 > 0 and y1 < sizeY and y2 > 0 and y2 < sizeY):
            frameLines.append(Line(int(round(x1)), int(round(y1)),
                                   int(round(x2)), int(round(y2))))
        else:
            y1 = 0
            x1 = -c/a

            y2 = sizeY
            x2 = -(c + b*y2) / a

            frameLines.append(Line(int(round(x1)), int(round(y1)),
                                   int(round(x2)), int(round(y2))))

    return frameLines
