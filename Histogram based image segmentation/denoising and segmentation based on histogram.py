# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:14:49 2023

@author: sagar
"""

'''  DENOISING AND SEGMENTATION BASED ON HISTOGRAM '''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,img_as_ubyte,img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma


image_path="D:\\Codes\\image processing\\Histogram based image segmentation\\test.jpg"
image=img_as_float(io.imread(image_path,as_gray=True))

print(image.shape)

patch_kw=dict(patch_size=5, patch_distance=6, multichannel=False)
sigma_est=np.mean(estimate_sigma(image, multichannel=False))
denoise=denoise_nl_means(image, h=1.15*sigma_est, fast_mode=True, **patch_kw)

denoise_ubyte=img_as_ubyte(denoise)

plt.imshow(denoise_ubyte)
plt.set_cmap('gray')
plt.axis('off')
plt.show()


plt.hist(img_as_ubyte(image).flat,bins=100, range=(0,255))
plt.title("Histogram of the Grayscale of main image")
plt.show()


plt.hist(denoise_ubyte.flat, bins=100, range=(0,255))
plt.title("Histogram of the denoised image")
plt.show()

segm1=(denoise_ubyte <= 64)
segm2=(denoise_ubyte <= 128) & (denoise_ubyte>64)
segm3=(denoise_ubyte <=192)  & (denoise_ubyte>128)
segm4=(denoise_ubyte>192)

print(segm1)

all_segments=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))

all_segments[segm1]=[1,0,0]    # showing all segments
all_segments[segm2]=[0,1,0]
all_segments[segm3]=[0,0,1]
all_segments[segm4]=[1,1,0]

plt.imshow(all_segments)
plt.axis('off')
plt.show()


all_segments[segm1]=[1,0,0]    # showing segment 1
all_segments[segm2]=[0,0,0]
all_segments[segm3]=[0,0,0]
all_segments[segm4]=[0,0,0]

plt.imshow(all_segments)
plt.axis('off')
plt.show()


all_segments[segm1]=[0,0,0]    # shoeing segment 2
all_segments[segm2]=[0,1,0]
all_segments[segm3]=[0,0,0]
all_segments[segm4]=[0,0,0]

plt.imshow(all_segments)
plt.axis('off')
plt.show()


all_segments[segm1]=[0,0,0]   # showing segment 3
all_segments[segm2]=[0,0,0]
all_segments[segm3]=[0,0,1]
all_segments[segm4]=[0,0,0]

plt.imshow(all_segments)
plt.axis('off')
plt.show()


all_segments[segm1]=[0,0,0]   # showing segment 4
all_segments[segm2]=[0,0,0]
all_segments[segm3]=[0,0,0]
all_segments[segm4]=[1,1,0]

plt.imshow(all_segments)
plt.axis('off')
plt.show()