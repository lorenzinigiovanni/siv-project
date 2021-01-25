from sklearn.cluster import KMeans
from cv2 import cv2 as cv2
import numpy as np

def sklearnKMeans(image):
    image = image.copy()

    image = cv2.blur(image, (5, 5))
    image = image / 255
    
    pic_n = image.reshape(
        image.shape[0] * image.shape[1], image.shape[2])

    kmeans = KMeans(n_clusters=3, random_state=0, tol=1e-3).fit(pic_n)
    pic2show = pic_n.copy() * 0

    for i, s in enumerate(kmeans.labels_):
        if(s == 0):
            pic2show[i][0] = 255
            pic2show[i][1] = 0
            pic2show[i][2] = 0
        elif(s == 1):
            pic2show[i][0] = 0
            pic2show[i][1] = 255
            pic2show[i][2] = 0
        elif(s == 2):
            pic2show[i][0] = 0
            pic2show[i][1] = 0
            pic2show[i][2] = 255

    res = pic2show.reshape(image.shape)
    return res

def cv2KMeans(image):
    image = image.copy()

    image = cv2.blur(image, (5, 5))

    Z = image.reshape(
        image.shape[0] * image.shape[1], image.shape[2])
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 2, 1.0)
    K = 3
    _, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    pic2show = Z.copy() * 0

    for i, s in enumerate(label.flatten()):
        if(s == 0):
            pic2show[i][0] = 255
            pic2show[i][1] = 0
            pic2show[i][2] = 0
        elif(s == 1):
            pic2show[i][0] = 0
            pic2show[i][1] = 255
            pic2show[i][2] = 0
        elif(s == 2):
            pic2show[i][0] = 0
            pic2show[i][1] = 0
            pic2show[i][2] = 255

    res = pic2show.reshape(image.shape)
    return res
