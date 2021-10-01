"""
% Before trying to construct hybrid images, it is suggested that you
% implement my_imfilter.m and then debug it using proj1_test_filtering.m

% Debugging tip: You can split your MATLAB code into cells using "%%"
% comments. The cell containing the cursor has a light yellow background,
% and you can press Ctrl+Enter to run just the code in that cell. This is
% useful when projects get more complex and slow to rerun from scratch
"""



#%% close all figures
import matplotlib.pyplot as plt
import os
import cv2
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage import img_as_float
from skimage.transform import rescale, resize, downscale_local_mean
plt.close('all') # closes all figures
from vis_hybrid_image import vis_hybrid_image


from my_imfilter import my_imfilter



def gaussian_filter(shape=(3,3),sigma=0.5):
    # for both dimensions sigma is taken to be same

   # this is replacement for the matlab's fspecial. 
    
    m=shape[0]//2
    n=shape[1]//2
    y,x = np.ogrid[-m:m+1,-n:n+1] # first time encountered in the internet explaination
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h # read from internet





def normalize(img):
    min = img.min()
    max = img.max()
    return ((img -min) / max - min)  ## read from internet



print(gaussian_filter((3, 3)))

#%% Setup
#% read images and convert to floating point format
image1 = mpimg.imread('../data/dog.bmp')
image2 = mpimg.imread('../data/cat.bmp')
image1= img_as_float(image1)
image2= img_as_float(image2)

"""
% Several additional test cases are provided for you, but feel free to make
% your own (you'll need to align the images in a photo editor such as
% Photoshop). The hybrid images will differ depending on which image you
% assign as image1 (which will provide the low frequencies) and which image
% you asign as image2 (which will provide the high frequencies)
"""

""" %% Filtering and Hybrid Image construction """
cutoff_frequency = 7  

"""This is the standard deviation, in pixels, of the 
% Gaussian blur that will remove the high frequencies from one image and 
% remove the low frequencies from another image (by subtracting a blurred
% version from the original version). You will want to tune this for every
% image pair to get the best results. """
gaussian_dim= cutoff_frequency*4 +1

filter = gaussian_filter((gaussian_dim, gaussian_dim), cutoff_frequency)
#insert values from fspecial('Gaussian', cutoff_frequency*4+1, cutoff_frequency) here

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE BELOW. Use my_imfilter to create 'low_frequencies' and
% 'high_frequencies' and then combine them to create 'hybrid_image'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove the high frequencies from image1 by blurring it. The amount of
% blur that works best will vary with different image pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
"""

low_frequencies = my_imfilter(image1, filter)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove the low frequencies from image2. The easiest way to do this is to
% subtract a blurred version of image2 from the original version of image2.
% This will give you an image centered at zero with negative values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

high_frequencies = image2 - my_imfilter(image2, filter)



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combine the high frequencies and low frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
hybrid_image =  normalize(low_frequencies + high_frequencies)


#%% Visualize and save outputs

plt.figure(1)
plt.imshow(normalize(low_frequencies))
plt.figure(2)

plt.imshow(normalize(high_frequencies + 0.5))

vis = vis_hybrid_image(hybrid_image) 
#see function script vis_hybrid_image.py
plt.figure(3)
plt.imshow(vis)
mpimg.imsave('../Results/low_frequencies.jpg',normalize(low_frequencies))
mpimg.imsave('../Results/high_frequencies.jpg',normalize(high_frequencies + 0.5))
mpimg.imsave('../Results/hybrid_image.jpg',hybrid_image)
mpimg.imsave('../Results/hybrid_image_scales.jpg',vis)
# %%
