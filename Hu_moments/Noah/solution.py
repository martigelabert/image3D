

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

filename = "../data/simple_shapes.png"
im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
_, bin_image = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# create an empty image for contours
img_contours = np.zeros(im.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 2)

Hu = np.zeros((7,1))
for n in range(len(contours)):
   x, y, w, h = cv2.boundingRect(contours[n])
   cv2.putText(img_contours, str(n), (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
   moments = cv2.moments(contours[n])
   huMoments = cv2.HuMoments(moments)
   for i in range(0, 7):
      if huMoments[i] != 0:
         huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(np.abs(huMoments[i]))
      else:
         huMoments[i] = 0
   Hu = np.c_[Hu, huMoments]
Hu = Hu[:, 1:] # Eliminate first column that is blank
# Each column of Hu corresponds to the 7 huMoments of each object

# We only need Hu0
ind_circ = np.argsort(Hu[0, :])[12:16]
ind_tri = np.argsort(Hu[0, :])[4: 8]
ind_rec = np.argsort(Hu[0, :])[0:4]
ind_sq = np.argsort(Hu[0, :])[8:12]
print('The circles are:' + str(ind_circ))
print('The triangles are:' + str(ind_tri))
print('The rectangles are:' + str(ind_rec))
print('The squares are:' + str(ind_sq))

plt.imshow(img_contours, cmap='gray')
plt.show()


