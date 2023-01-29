# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:36:35 2023

@author: mrsag
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import io,measure,color
from skimage.transform import resize


# Construct some test data
image_path="D:\\Codes\\image processing\\BeFunky-collage.jpg"    
image=io.imread(image_path)          #opening the image
#r=color.rgba2rgb(r)                 # converting 4 colour to 3 colour
image=color.rgb2gray(image)          # converting 3 colour image to greyscale
image=resize(image,(500,500))

#image=image[10:(image.shape[0]-10),10:(image.shape[1]-10)]  # when the image has unnecessary margins
# Find contours at a constant value of 0.8
contours = measure.find_contours(image, 0.2)    # finding contours at proper contrast value
print(np.array(contours).shape)

# Display the image and plot all contours found


fig, ax = plt.subplots()    
ax.imshow(image, cmap=plt.cm.gray)

for i in range(np.array(contours).shape[0]):
    ax.plot(contours[i][:,1],contours[i][:,0],linewidth=1)
    plt.text(contours[i][0][0],contours[i][0][1],'%d'%i,color='yellow')
    
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

#print(image.shape)
