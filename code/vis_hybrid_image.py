import matplotlib.pyplot as plt
import os
import cv2
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
from numpy import concatenate as cat
def  vis_hybrid_image(hybrid_image):

  """
  %visualize a hybrid image by progressively downsampling the image and
  %concatenating all of the images together.
  """
  scales = 5#how many downsampled versions to create
  padding = 5 #how many pixels to pad.

  original_height = hybrid_image.shape[0]
  num_colors = hybrid_image.shape[2]
  #counting how many color channels the input has
  output = hybrid_image
  cur_image = hybrid_image

  for i in range(2,scales+1):

      gap_array= np.ones((original_height, padding, num_colors))

      output = cat((output,gap_array), axis = 1) # wrongly done in the sample code

    
      cur_image = resize(cur_image,(cur_image.shape[0] // 2, cur_image.shape[1] // 2), anti_aliasing=True)

      vertical_gap= np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors))
      tmp = cat((vertical_gap, cur_image), axis =0)
      output = cat( (output, tmp),axis=1)

  return(output)