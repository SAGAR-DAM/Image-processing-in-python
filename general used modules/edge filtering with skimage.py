# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:40:25 2023

IMAGE EDGE FILTERING

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny

# Load the image
image_path='D:\\pictures\\5.png'
image = img_as_ubyte(io.imread(image_path,as_gray=True))

image_roberts=roberts(image)
image_sobel=sobel(image)
image_scharr=scharr(image)
image_prewitt=prewitt(image)
image_canny=canny(image, sigma=0.5)

print("Image shape: ",image.shape)
print("Roberts image shape: ",image_roberts.shape)
print("Sobel image shape: ",image_sobel.shape)
print("Scharr image shape: ",image_scharr.shape)
print("Prewitt image shape: ",image_prewitt.shape)
print("Canny image shape: ",image_canny.shape)

plt.figure()
plt.axis('off')
plt.set_cmap('gray')
#plt.imshow(image)
#plt.imshow(image_roberts)
plt.imshow(image_sobel)
#plt.imshow(image_scharr)
#plt.imshow(image_prewitt)
#plt.imshow(image_canny)
