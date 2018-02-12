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
    slopes = [ (line[0][3]-line[0][1])/(line[0][2]-line[0][0]) for line in lines]
    rights = [ [line, slope, line[0][1] - slope*line[0][0]] for line,slope in zip(lines, slopes) if slope > 0.5 and slope < 1.0 ] # and not np.isnan(slope) ]
    lefts  = [ [line, slope, line[0][1] - slope*line[0][0]] for line,slope in zip(lines, slopes) if slope < -0.5 and slope > -1.0 ] # and slope > -0.5 and not np.isnan(slope) ]
    #lefts[0] = [ [[x1,y1,x2,y2]] , slope , y_intercept ]

    y_mins = [ min(line[0][1],line[0][3]) for line in lines]
    y_min  = min(y_mins)
    y_max  = img.shape[0]

    for lanes in [rights,lefts]:
        slope_mean = np.mean( [ lane[1] for lane in lanes ] )
        slope_std  = np.std ( [ lane[1] for lane in lanes ] )
        if slope_std == 0:
            slope = slope_mean
        else:
            slope = np.mean( [ lane[1] for lane in lanes if lane[1] - slope_mean < 2*slope_std ] )            
        print()
        print('slope : {}'.format(slope))

        intercept_mean = np.mean( [ lane[2] for lane in lanes ] )
        intercept_std  = np.std ( [ lane[2] for lane in lanes ] )
        if intercept_std == 0:
            intercept = intercept_mean
        else:
            intercept = np.mean( [ lane[2] for lane in lanes if lane[2] - intercept_mean < 2*intercept_std ] )
        print('intercept : {}'.format(intercept))
        
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