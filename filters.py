import cv2 as cv
import numpy as np
import copy

def applyFilters(image, filterList):
	grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

	while len(filterList) > 0:
		newFilter = filterList.pop(0)
		
		if newFilter == 'edgepreserve':
			image = cv.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
		
		if newFilter == 'pencilsketch':
			dst_gray, image = cv.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
			
		if newFilter == 'water':
			image = cv.stylization(image, sigma_s=60, sigma_r=0.07)
			
		if newFilter == 'grayscale':
			image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			
		if newFilter == 'blackwhite':
			(thresh, image) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
			
		if newFilter == 'jet':
			image = cv.applyColorMap(grayImage, cv.COLORMAP_JET)
		
		if newFilter == 'hot':
			image = cv.applyColorMap(grayImage, cv.COLORMAP_HOT)
		
		if newFilter == 'sepia':
			i, j, k = image.shape
			originalImage = copy.deepcopy(image)
			
			for x in range(i):
				for y in range(j):
					R = originalImage[x,y,2] * 0.393 + originalImage[x,y,1] * 0.769 + originalImage[x,y,0] * 0.189
					G = originalImage[x,y,2] * 0.349 + originalImage[x,y,1] * 0.686 + originalImage[x,y,0] * 0.168
					B = originalImage[x,y,2] * 0.272 + originalImage[x,y,1] * 0.534 + originalImage[x,y,0] * 0.131
					
					if R > 255:
						image[x,y,2] = 255
					else:
						image[x,y,2] = R
         
					if G > 255:
						image[x,y,1] = 255
					else:
						image[x,y,1] = G
         
					if B > 255:
						image[x,y,0] = 255
					else:
						image[x,y,0] = B
	
		if newFilter == 'enhance':
			image = cv.detailEnhance(image)
			
		if newFilter == 'blur':
			image = cv.blur(image, (10, 10))
			
		if newFilter == 'blue':
			blueImage = np.full(image.shape, (255,0,0), np.uint8)
			image = cv.addWeighted(image, 0.75, blueImage, 0.25, 0)
		
		if newFilter == 'green':
			greenImage = np.full(image.shape, (0,255,0), np.uint8)
			image = cv.addWeighted(image, 0.75, greenImage, 0.25, 0)
		
		if newFilter == 'red':
			redImage = np.full(image.shape, (0,0,255), np.uint8)
			image = cv.addWeighted(image, 0.75, redImage, 0.25, 0)
		
	return image
	
#image = cv.imread('christinaaguilera.jpg') # Importing Sample Test Image
	
#image = applyFilters(image, ['red', 'blur'])

#cv.imwrite('./final.jpg', image)



