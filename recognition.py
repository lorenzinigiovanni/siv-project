from cv2 import cv2
from utils import getColorMean, getHistogram


def recognitionByColor(sixths, beeColour, openCellColour, closeCellColour):
    results = []

    for sixth in sixths:
        mean = getColorMean(sixth)

        beeDifference = [e - mean[c] for c, e in enumerate(beeColour)]
        openCellDifference = [e - mean[c]
                              for c, e in enumerate(openCellColour)]
        closeCellDifference = [e - mean[c]
                               for c, e in enumerate(closeCellColour)]

        beeSquare = beeDifference[0] ** 2 + \
            beeDifference[1] ** 2 + beeDifference[2] ** 2
        openCellSquare = openCellDifference[0] ** 2 + \
            openCellDifference[1] ** 2 + openCellDifference[2] ** 2
        closeCellSquare = closeCellDifference[0] ** 2 + \
            closeCellDifference[1] ** 2 + closeCellDifference[2] ** 2

        if (beeSquare < openCellSquare and beeSquare < closeCellSquare):
            results.append(1)
        elif (openCellSquare < beeSquare and openCellSquare < closeCellSquare):
            results.append(2)
        elif (closeCellSquare < beeSquare and closeCellSquare < openCellSquare):
            results.append(3)

    return results


def recognitionByHistogram(sixths, beeHistogram, openCellHistogram, closedCellHistogram):
    results = []

    for sixth in sixths:
        meanHistogram = getHistogram(sixth)

        beeHistDifference = cv2.compareHist(
            beeHistogram, meanHistogram, cv2.HISTCMP_CORREL)
        openCellHistDifference = cv2.compareHist(
            openCellHistogram, meanHistogram, cv2.HISTCMP_CORREL)
        closeCellHistDifference = cv2.compareHist(
            closedCellHistogram, meanHistogram, cv2.HISTCMP_CORREL)

        if (beeHistDifference > openCellHistDifference and beeHistDifference > closeCellHistDifference):
            results.append(1)
        elif (openCellHistDifference > beeHistDifference and openCellHistDifference > closeCellHistDifference):
            results.append(2)
        elif (closeCellHistDifference > beeHistDifference and closeCellHistDifference > openCellHistDifference):
            results.append(3)

    return results
