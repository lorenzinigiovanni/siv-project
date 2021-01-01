from cv2 import cv2 as cv2
import numpy as np


class MousePoints:
    def __init__(self, windowname, img):
        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(windowname, img)
        self.curr_pt = []
        self.point = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x, y]

    def getpt(self, count=1, img=None):
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.img)
        cv2.setMouseCallback(self.windowname, self.select_point)
        self.point = []
        while(1):
            cv2.imshow(self.windowname, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point) >= count:
                break
        cv2.setMouseCallback(self.windowname, lambda *args: None)
        return self.point, self.img
