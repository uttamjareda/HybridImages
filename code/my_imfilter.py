### done fully

"""
% This function that you will implement is intended to behave like the built-in function 
% imfilter() in Matlab or equivalently the same function implemented as part of scipy.misc module
% in Python. You will implement imfilter from first principles, i.e., without using 
% any library functions. 

% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, we will essentially perform 2D correlation 
% between the filter and image. Referring to 'proj1_test_filtering.py' would help you with
% your implementation. 
  
% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' in Matlab, you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach would be to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call library imfilter so you can see the desired
% % behavior. When you write your actual solution, **you can't use imfilter,
% % correlate, convolve commands, but implement the same using matrix manipulations**. 
% % Simply loop over all the pixels and do the actual
% % computation. It might be slow.
"""



import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
""" Exemplar Gaussian 3x3 filter shown below-- see filters defined in proj1_test_filtering.py """
filter = np.asarray([[0.1019,0.1154,0.1019],[0.1154,0.1308,0.1154],[0.1019,0.1154,0.1019]],dtype=np.float32) 

def my_imfilter( image, filter):
   
    m=filter.shape[0]
    n=filter.shape[1]
   
    img_dim=image.shape
    
    
    if(m%2==1 and n%2==1):
        
        filtered_image= image.copy()
        padded_row_num= int( img_dim[0] + 2*int((m/2)))
        padded_col_num= int(img_dim[1] + 2*int (n/2))
       
        

        if(image[0][0].size!=1):
            padded_array=np.zeros((padded_row_num, padded_col_num, image[0][0].size))
            padded_array[m//2:(m//2+img_dim[0]), n//2: (n//2+img_dim[1])] = image

            for i in range(img_dim[2]):
                for x in range (img_dim[0]):
                    for y in range (img_dim[1]):
                        filtered_image[x][y][i] = sum(sum(filter * padded_array[x:x+m , y:y+n, i]))
            return filtered_image

        else:
            padded_array=np.zeros((padded_row_num,padded_col_num))
            padded_array[m//2:(m//2+img_dim[0]), n//2: (n//2+img_dim[1])] = image

            for x in range  (img_dim[0]):
                for y in range (img_dim[1]):
                    filtered_image[x][y] = sum(sum(filter * padded_array[x:x+m , y:y+n]))
            

            return filtered_image


            
    else:
        print("Please specify correct sized filter with both dimensions as odd")
        return ;




img=Image.open("/home/taneesh/5thSem/Computer Vision/Assignment-1/Assignment1/data/bicycle.bmp")
img=np.asarray(img)

#newimg= my_imfilter(img, filter)
plt.imshow(img)

