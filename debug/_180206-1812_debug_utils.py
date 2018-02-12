
# coding: utf-8
# In[1]: Import Packages -------------------------------------------
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from IPython import get_ipython
ipython = get_ipython()
#from IPython.display import HTML
import argparse

#from topLevel import draw_lines
from debug import draw_lines

from datetime import datetime as dt
import time
#get_ipython().mag

# Parameter
default_dir   = 'C:/Users/mo/home/_eSDC2_/_PRJ01_/_2_WIP/_161207-0108_Submitted/test_images/'
default_video = 'C:/Users/mo/home/_eSDC2_/_PRJ01_/_2_WIP/_161207-0108_Submitted/test_videos/'
default_path  = 'C:/Users/mo/home/_eSDC2_/_PRJ01_/_2_WIP/_161207-0108_Submitted/mo/'

# default_threshold    = (60, 45, 180) # (50, 45, 150) # (low_threshold, threshold, high_threshold)

# Helper function(s): command-line / parse parameters
def parse_args():
    parser = argparse.ArgumentParser(description='finding lane lines')
    parser.add_argument('-d', '--dir', dest='dir', help='root directory path', action='store', type=str, default=default_dir)
    parser.add_argument('-p', '--path', dest='path', help='logs directory path', action='store', type=str, default=default_path)
    parser.add_argument('-v', '--video', dest='video', help='video directory path', action='store', type=str, default=default_video)
    parser.add_argument('-t', '--threshold', help='tuple = (low_threshold, threshold, high_threshold)', dest='threshold', type=tuple, default=(60, 45, 180)) # canny()
    parser.add_argument('-k', '--kernel', help='kernel size', dest='kernel_size', type=int, default=3) # gaussian_blur()
    parser.add_argument('-r', '--rho', help='rho', dest='rho', type=int, default=2) # hough_lines()
    parser.add_argument('-a', '--theta', help='theta', dest='theta', type=float, default=np.pi/180) # hough_lines()
    parser.add_argument('-l', '--len', help='min_line_len', dest='min_line_len', type=int, default=50) # hough_lines()
    parser.add_argument('-g', '--gap', help='max_line_gap', dest='max_line_gap', type=int, default=150) # hough_lines()
    args   = parser.parse_args()
    return args

args = parse_args()


def grayscale(img):
    '''Applies the Grayscale transform. NOTE: call plt.imshow(gray, cmap='gray')'''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  

def canny(args, img):
    '''Applies the Canny transform'''
    return cv2.Canny(img, args.threshold[0], args.threshold[2])


def gaussian_blur(args, img):
    '''Applies a Gaussian Noise kernel'''
    return cv2.GaussianBlur(img, (args.kernel_size, args.kernel_size), 0)


def region_of_interest(img, vertices):
    '''
    Applies an image mask. Only keeps the region of the image. The rest of the image is set to black.
    '''
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by 'vertices' with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    '''
    - `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    - `initial_img` should be the image before any processing.
    - The result image is computed as follows: initial_img * α + img * β + λ
    - NOTE: initial_img and img must be the same shape!
    '''
    return cv2.addWeighted(initial_img, α, img, β, λ)


def hough_lines(args, img):
    '''
    `img` should be the output of a Canny transform. Returns an image with hough lines drawn.
    '''
    lines = cv2.HoughLinesP(img, args.rho, args.theta, args.threshold[1], np.array([]), args.min_line_len, args.max_line_gap) # minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def save_figure(img):
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    file_name  = dt.now().strftime('%y%m%d_%H%M_%S.%f')[:-4]
    fig.savefig(args.path+file_name+'.jpg')
    plt.close(fig)
    print('')


def find_lines(args, image): # def FindingLaneLinesImg(args, image):
    imshape = image.shape
    gray = grayscale(image)  #Finding Lane Lines
    blur_gray = gaussian_blur(args, gray)       
    edges = canny(args, blur_gray)  #Identify the edges with the Canny fct      
    vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)  #Define a four sided polygon to mask
    masked_edges = region_of_interest(edges, vertices)  #Create a masked edges image using cv2.fillPoly()
    hough_lines0 = hough_lines(args, masked_edges) #Hough transform # note: goo.gl/eL1kU4
    weighted_img0 = weighted_img(hough_lines0, image, α=0.8, β=1., λ=0.) #Draw the lines on the edge image
    save_figure(weighted_img0)

    return weighted_img0


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    #Import packages
    result = find_lines(args, image)
       
    return result


def video_white(args):
    white_output = args.path+'white.mp4'
    clip1 = VideoFileClip(args.video+'solidWhiteRight.mp4')
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')    


def video_yellow(args):
    yellow_output = args.path+'yellow.mp4'
    clip2 = VideoFileClip(args.video+'solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


def video_extra(args):
    challenge_output = 'extra.mp4'
    clip2 = VideoFileClip('challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


'''
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
'''
