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
    y_mins, y_max      = [], img.shape[0]
    slopes, intercepts = [], []

    lanes = {}
    lanes['right'], lanes['left']  = {}, {}
    lanes['right']['grades'], lanes['right']['y_intercepts'] = [], []
    lanes['left']['grades'], lanes['left']['y_intercepts'] = [], []
    
    for line in lines:
        slopes.append( ( line[0][3] - line[0][1] ) / ( line[0][2] - line[0][0] ) )
        intercepts.append( line[0][1] - slopes[-1] * line[0][0] )
        y_mins.append( min( line[0][1], line[0][3] ) )

        if slopes[-1] > 0.5 and slopes[-1] < 1.0:
            lane_lines['right']['grades'].append(slopes[-1]) # right_slopes.append(slopes[-1])
            lane_lines['right']['y_intercepts'].append(intercepts[-1]) # right_intercepts.append(intercepts[-1])
        elif slopes[-1] < -0.5 and slopes[-1] > -1.0:
            lane_lines['left']['grades'].append(slopes[-1]) # left_slopes.append(slopes[-1])
            lane_lines['left']['y_intercepts'].append(intercepts[-1]) # left_intercepts.append(intercepts[-1])

    y_min  = min(y_mins)

    for lane_line in lane_lines:
        slope_mean, intercept_mean = np.mean( lane_line['grades'] ), np.mean( lane_line['y_intercepts'] )
        slope_std,  intercept_std  = np.std ( lane_line['grades'] ), np.std ( lane_line['y_intercepts'] )
        slope, intercept = [ slope,intercept for slope,intercept in zip( lane_line['grades'], lane_line['y_intercepts'] ) if slope - slope_mean < 2*slope_std ]
     
        x_min     = int( ( y_min - intercept ) / slope ) 
        x_max     = int( ( y_max - intercept ) / slope )

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