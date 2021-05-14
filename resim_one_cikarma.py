import cv2
import numpy as np
from matplotlib import  pyplot as plt

resim = cv2.imread('on_plan.jpg')

mask = np.zeros(resim.shape[:2], np.uint8)#ilk iki elemanı alır

bgdModel = np.zeros((1,65),np.float64)#bu iki parametre algoritmaz tarafında alınmaktadır.
fgdModel = np.zeros((1,65),np.float64)

dikdortgen = (250,125,150,250)#alınacak yerin koordinatları

cv2.grabCut(resim,mask,dikdortgen,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0,1).astype('uint8')
resim *= mask2[:,:,np.newaxis]

plt.imshow(resim)
plt.colorbar()
plt.show()