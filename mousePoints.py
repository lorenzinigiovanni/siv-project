from cv2 import cv2 as cv2
import numpy as np


class MousePoints:
    def __init__(self, windowname, img, boxDim=0):
        self.windowname = windowname
        self.img = img.copy()
        self.img2 = img.copy()
        self.point = []
        self.boxDim = boxDim

    def select_point(self, event, x, y, flags, param):
        self.img = self.img2.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])
            cv2.circle(self.img2, (x, y), 5, (0, 255, 0), -1)
            self.img = self.img2
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.point.append([-1, -1])
            self.img = self.img2
        elif event == cv2.EVENT_MOUSEMOVE and self.boxDim > 0:
            cv2.rectangle(self.img, (round(x - self.boxDim/2), round(y - self.boxDim/2)),
                          (round(x + self.boxDim / 2), round(y + self.boxDim / 2)), (255, 255, 255), 1)
            cv2.imshow(self.windowname, self.img)
        elif event == cv2.EVENT_MOUSEWHEEL and self.boxDim > 0:
            if flags > 0:
                self.boxDim += 2
            else:
                self.boxDim -= 2

            if self.boxDim < 2:
                self.boxDim = 2
            elif self.boxDim > self.img.shape[0]:
                self.boxDim = self.img.shape[0]

            cv2.rectangle(self.img, (round(x - self.boxDim/2), round(y - self.boxDim/2)),
                          (round(x + self.boxDim / 2), round(y + self.boxDim / 2)), (255, 255, 255), 1)
            cv2.imshow(self.windowname, self.img)

    def getpt(self, count=1):
        cv2.namedWindow(self.windowname, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.windowname, self.img)
        cv2.setMouseCallback(self.windowname, self.select_point)

        self.point = []
        while(1):
            cv2.imshow(self.windowname, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point) >= count:
                break

        cv2.setMouseCallback(self.windowname, lambda *args: None)

        return self.point, self.boxDim
