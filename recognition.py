from cv2 import cv2
import math

from utils import getColorMean, getHistogram


def recognitionByColor(sixths, beeColour, openCellColour, closeCellColour):
    results = []

    for sixth in sixths:
        mean = getColorMean(sixth)

        beeSquare = math.inf
        openCellSquare = math.inf
        closeCellSquare = math.inf

        if beeColour is not [-1, -1, -1]:
            beeDifference = [e - mean[c] for c, e in enumerate(beeColour)]
            beeSquare = beeDifference[0] ** 2 + \
                beeDifference[1] ** 2 + beeDifference[2] ** 2

        if openCellColour is not [-1, -1, -1]:
            openCellDifference = [e - mean[c]
                                  for c, e in enumerate(openCellColour)]
            openCellSquare = openCellDifference[0] ** 2 + \
                openCellDifference[1] ** 2 + openCellDifference[2] ** 2

        if closeCellColour is not [-1, -1, -1]:
            closeCellDifference = [e - mean[c]
                                   for c, e in enumerate(closeCellColour)]
            closeCellSquare = closeCellDifference[0] ** 2 + \
                closeCellDifference[1] ** 2 + closeCellDifference[2] ** 2

        if (beeSquare < openCellSquare and beeSquare < closeCellSquare):
            results.append("Bee")
        elif (openCellSquare < beeSquare and openCellSquare < closeCellSquare):
            results.append("Open")
        elif (closeCellSquare < beeSquare and closeCellSquare < openCellSquare):
            results.append("Close")

    return results


def recognitionByHistogram(sixths, beeHistogram, openCellHistogram, closeCellHistogram):
    results = []

    for sixth in sixths:
        meanHistogram = getHistogram(sixth)

        beeHistCorrelation = 0
        openCellHistCorrelation = 0
        closeCellHistCorrelation = 0

        if beeHistogram is not None:
            beeHistCorrelation = cv2.compareHist(
                beeHistogram, meanHistogram, cv2.HISTCMP_CORREL)
        if openCellHistogram is not None:
            openCellHistCorrelation = cv2.compareHist(
                openCellHistogram, meanHistogram, cv2.HISTCMP_CORREL)
        if closeCellHistogram is not None:
            closeCellHistCorrelation = cv2.compareHist(
                closeCellHistogram, meanHistogram, cv2.HISTCMP_CORREL)

        if (beeHistCorrelation > openCellHistCorrelation and beeHistCorrelation > closeCellHistCorrelation):
            results.append("Bee")
        elif (openCellHistCorrelation > beeHistCorrelation and openCellHistCorrelation > closeCellHistCorrelation):
            results.append("Open")
        elif (closeCellHistCorrelation > beeHistCorrelation and closeCellHistCorrelation > openCellHistCorrelation):
            results.append("Close")

    return results
