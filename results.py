from cv2 import cv2


def showResults(results, title, image, sizeX, sizeY):
    print(title)

    numberOfAreas = sizeX * sizeY

    bee = results.count("Bee")
    open = results.count("Open")
    close = results.count("Close")

    print("Bee: " + str(bee) + "/" + str(numberOfAreas) + ", Open: " +
          str(open) + "/" + str(numberOfAreas) + ", Close: " + str(close) + "/" + str(numberOfAreas))

    print("Bee: " + str(round(100 * bee / numberOfAreas)) + "%, Open: " +
          str(round(100 * open / numberOfAreas)) + "%, Close: " + str(round(100 * close / numberOfAreas)) + "%")

    overlay = image.copy() * 0

    counter = 0

    for yy in range(0, sizeY):
        for xx in range(0, sizeX):
            x = image.shape[1]
            y = image.shape[0]

            xStart = round(xx * (x / sizeX))
            yStart = round(yy * (y / sizeY))
            xEnd = round((1 + xx) * (x / sizeX))
            yEnd = round((1 + yy) * (y / sizeY))

            color = (0, 0, 0)

            if (results[counter] == "Bee"):
                color = (255, 0, 0)
            elif (results[counter] == "Open"):
                color = (0, 255, 0)
            elif (results[counter] == "Close"):
                color = (0, 0, 255)

            cv2.rectangle(overlay, (xStart, yStart),
                          (xEnd, yEnd), color, -1)

            if (numberOfAreas <= 50):
                cv2.rectangle(overlay, (xStart, yStart),
                              (xEnd, yEnd), (255, 255, 255), 2)
                cv2.putText(
                    overlay, results[counter], (xStart + 10, yEnd - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            counter += 1

    res = cv2.addWeighted(image, 0.7, overlay, 0.3, 1.0)
    cv2.imshow(title, res)
