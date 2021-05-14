import cv2
import numpy as np
import matplotlib.pyplot as plt

aranacak_resim = cv2.imread('kucuk_resim.JPG',0)
buyuk_resim = cv2.imread('buyuk_resim.JPG',0)

orb = cv2.ORB_create()

key_point1, hedef1 = orb.detectAndCompute(aranacak_resim, None)
key_point2, hedef2 = orb.detectAndCompute(buyuk_resim, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
eslesmeler = bf.match(hedef1,hedef2)
eslesmeler = sorted(eslesmeler, key=lambda x:x.distance)
son_resim =cv2.drawMatches(aranacak_resim,key_point1,buyuk_resim,key_point2,eslesmeler[:10],None,flags=2)
plt.imshow(son_resim)
plt.show()





