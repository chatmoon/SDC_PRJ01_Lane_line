# coding: utf-8
# In[1]: Import Packages -------------------------------------------
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from IPython import get_ipython
ipython = get_ipython()

#from IPython.display import HTML
import numpy as np
import cv2
import math
import pandas as pd
import os.path
#get_ipython().magic('matplotlib inline')
from debug_utils import * 


def draw_lines(args, img, lines, color=[255, 0, 0], thickness=8):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    """
    y_mins, y_max = [], img.shape[0]

    slopes = {}
    slopes['right'], slopes['left'] = [], []
    
    for line in lines:
        slope = ( line[0][3] - line[0][1] ) / ( line[0][2] - line[0][0] )
        y_mins.append( min( line[0][1], line[0][3] ) )

        if slope > 0.5 and slope < 1.0:
            slopes['right'].append(slope)
        elif slope < -0.5 and slope > -1.0:
            slopes['left'].append(slope)

    y_min  = min(y_mins)

    for side in slopes.keys():      
        slope_mean  = np.mean( slopes[side] )
        slope_std   = np.std ( slopes[side] )
        slope       = np.mean( [ slope  for slope  in  slopes[side] if slope - slope_mean < 2*slope_std ] )
        y_intercept = 
     
        x_min = int( ( y_min - y_intercept ) / slope ) 
        x_max = int( ( y_max - y_intercept ) / slope )

        cv2.line(img, (x_min, y_min), (x_max, y_max), color, thickness)

def main():
    flags = parse_args()
    # files = os.listdir(flags.dir)

    # # Finding Lane Lines -------------------------------------------
    # for file in files:
    #     image =  cv2.imread(flags.dir+str(file)) # mpimg.imread(flags.dir+str(file));
    #     find_lines(flags, image)

    # video_white(flags)
    # video_yellow(flags)
    video_extra(flags)

if __name__ == '__main__':
    main()