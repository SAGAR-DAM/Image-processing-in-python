# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:40:25 2023

IMAGE DECONVOLUTION  

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage import restoration

# Load the image
image_path='D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\ids raw\\5_41.png'
image = img_as_ubyte(io.imread(image_path,as_gray=True))
psf=np.ones((3,3))/9
deconvolved, _ = restoration.unsupervised_wiener(image, psf)

plt.imshow(image)
plt.imshow(deconvolved)


