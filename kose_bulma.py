import numpy as np
import cv2

image = cv2.imread('kose_bulma.jpg')

griton = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
griton = np.float32(griton)
koseler = cv2.goodFeaturesToTrack(griton,300,0.01,10)#returns all the locations of the corners present in the gray scale image.
koseler = np.int0(koseler)

for kose in koseler:
    x, y = kose.ravel()
    cv2.circle(image,(x,y),3,255,-1)

cv2.imshow('koseler',image)
cv2.waitKey(0)
cv2.destroyAllWindows()