import cv2 as cv
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as pltplt

img = cv.imread('christinaaguilera.jpg') # Importing Sample Test Image


### Edge Preserving Filter #####
# edge = cv.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
#
# ### Pencil Sketch ########
# dst_gray, dst_color = cv.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
#
# ### Stilizare Water Color ######
# water = cv.stylization(img, sigma_s=60, sigma_r=0.07)

### Black and White ####
# im_gray = cv.imread('christinaaguilera.jpg', cv.IMREAD_GRAYSCALE)
# (thresh, im_bw) = cv.threshold(im_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# thresh = 127
#im_bw = cv.threshold(im_gray, thresh, 255, cv.THRESH_BINARY)[1]

### Just grayscale ###
#im_gray = cv.imread('christinaaguilera.jpg', cv.IMREAD_GRAYSCALE)

### Extra Saturated Colors
#jet_color = cv.applyColorMap(im_gray, cv.COLORMAP_JET)

### Culori Calde ###
#hot_color = cv.applyColorMap(im_gray, cv.COLORMAP_HOT)

### Sepia #####
# i, j, k = img.shape
# newImage = cv.imread('christinaaguilera.jpg')
# for x in range(i):
#     for y in range(j):
#         R = img[x,y,2] * 0.393 + img[x,y,1] * 0.769 + img[x,y,0] * 0.189
#         G = img[x,y,2] * 0.349 + img[x,y,1] * 0.686 + img[x,y,0] * 0.168
#         B = img[x,y,2] * 0.272 + img[x,y,1] * 0.534 + img[x,y,0] * 0.131
#         if R > 255:
#             newImage[x,y,2] = 255
#         else:
#             newImage[x,y,2] = R
#         if G > 255:
#             newImage[x,y,1] = 255
#         else:
#             newImage[x,y,1] = G
#         if B > 255:
#             newImage[x,y,0] = 255
#         else:
#             newImage[x,y,0] = B

### Enhance #####
enhance = cv.detailEnhance(img)


cv.imwrite('./final.jpg', enhance)



