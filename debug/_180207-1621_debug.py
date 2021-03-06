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
    rights = [ [line, slope, line[0][1] - slope*line[0][0]] for line,slope in zip(lines, slopes) if slope > 0.0 ] #  and slope < 0.5 and not np.isnan(slope) ]
    lefts  = [ [line, slope, line[0][1] - slope*line[0][0]] for line,slope in zip(lines, slopes) if slope < 0.0 ] # and slope > -0.5 and not np.isnan(slope) ]
    #lefts[0] = [ [[x1,y1,x2,y2]] , slope , y_intercept ]

    y_mins = [ min(line[0][1],line[0][3]) for line in lines]
    y_min  = min(y_mins)
    y_max  = img.shape[0]

    log_new = [slopes, rights, lefts, y_mins, y_min, y_max]

    for lanes in [rights,lefts]:
        slope     = np.mean( [ lane[1] for lane in lanes ] )
        log_new.append(slope)
        intercept = np.mean( [ lane[2] for lane in lanes ] )
        log_new.append(intercept)
        x_min     = int( ( y_min - intercept ) / slope ) 
        log_new.append(x_min)
        x_max     = int( ( y_max - intercept ) / slope )
        log_new.append(x_max)
        cv2.line(img, (x_min, y_min), (x_max, y_max), color, thickness)

    # update log: read csv_file.csv
    headers  = args.header
    log_line = pd.read_csv(args.path+args.csv_file, skiprows=[0], names=headers)
    df       = pd.DataFrame([ log_new ], columns=headers)

    # update log: add new entry into the log
    result = pd.concat([log_line, df], ignore_index=True)
    result.to_csv(args.path+args.csv_file) #, index=False)


    # update log: read csv_file.csv
    if os.path.isfile(args.path+args.csv_file):
        print()
        print('os.path.isfile: {}'.format(os.path.isfile(args.path+args.csv_file)))
        a = input('press key to continue')
        log_line = pd.read_csv(args.path+args.csv_file, skiprows=[0], names=args.header)
    else:
        print()
        print('os.path.isfile: {}'.format(os.path.isfile(args.path+args.csv_file)))
        a = input('press key to continue')
        log_line = pd.DataFrame([ ], columns=args.header)
    df = pd.DataFrame([ log_new ], columns=args.header)

    # update log: add new entry into the log
    result = pd.concat([log_line, df], ignore_index=True)
    result.to_csv(args.path+args.csv_file) #, index=False)


    # try: 
    #     log_line = pd.read_csv(args.path+args.csv_file, skiprows=[0], names=args.header)
    # except:
    #     log_line = pd.DataFrame([ ], columns=args.header)
    # finally:
    #     df       = pd.DataFrame([ log_new ], columns=args.header)
    #     # update log: add new entry into the log
    #     result = pd.concat([log_line, df], ignore_index=True)
    #     result.to_csv(args.path+args.csv_file) #, index=False)




def main():
    flags = parse_args()
    # files = os.listdir(flags.dir)

    # # Finding Lane Lines -------------------------------------------
    # for file in files:
    #     image =  cv2.imread(flags.dir+str(file)) # mpimg.imread(flags.dir+str(file));
    #     find_lines(flags, image)

    video_white(flags)
    # video_yellow(flags)

if __name__ == '__main__':
    main()