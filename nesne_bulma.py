import cv2
import numpy as np

imag_rgb = cv2.imread('ana_resim.jpg')
imggray = cv2.cvtColor(imag_rgb, cv2.COLOR_BGR2GRAY)

nesne = cv2.imread('template.jpg')
nesne = cv2.cvtColor(nesne,cv2.COLOR_BGR2GRAY)

w,h = nesne.shape[::-1]

resource = cv2.matchTemplate(imggray,nesne,cv2.TM_CCOEFF_NORMED)

threshold = 0.73

loc = np.where(resource>threshold)

for i in zip(*loc[::-1]):
    cv2.rectangle(imag_rgb, i, (i[0]+w, i[1]+h),(0,255,255),2)

cv2.imshow('nesne', imag_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()