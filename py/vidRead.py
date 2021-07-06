#!/usr/bin/env python
'''Functions for reading and analyzing shear droplet videos'''

# external packages
import cv2 as cv
from imutils.video import FPS
import imutils
import numpy as np 
import os
import sys
import time
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO
from matplotlib import pyplot as plt
import pandas as pd
import traceback

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from vidMorph import *
from vidCrop import *
from imshow import imshow

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for s in ['matplotlib', 'imageio', 'IPython', 'PIL']:
    logging.getLogger(s).setLevel(logging.WARNING)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__version__ = "1.0.0"
__maintainer__ = "Leanne Friedrich"
__email__ = "Leanne.Friedrich@nist.gov"
__status__ = "Development"


#----------------------------------------------


            

            
####### CROP TOOLS

def cropBlack(img:np.array, pad:int=10) -> np.array:
    '''crop black borders out of image. Designed for images with visible aperture. pad with extra pixels at end https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv'''
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(gray,1,255,cv.THRESH_BINARY)
    thresh = openMorph(thresh, 50) # clean up the image to make more black border
    _, contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv.boundingRect(cnt)
    if w<pad or h<pad:
        return ValueError('Cropped image is too small.')
    imh,imw = gray.shape
    bounds = padBounds(x,y,w,h,imh,imw,pad) # pad bounds of cropped image
    center = findCenter(x,y,w,h,imh,imw, bounds) # find center of aperture
    crop = imcrop(img, bounds)
    bounds = {**bounds, **center}
    return crop, bounds

            
############ FRAME READING TOOLS

def readFrames(file:str) -> None:
    '''iterate through frames'''

    stream = cv.VideoCapture(file) # open the file
    
    dropletTab = pd.dataFrame(cols=['time', 'dropNum', 'x', 'y', 'w', 'l', 'angle'])
    
#     while True:
    grabbed, frame = stream.read() # read first frame
    if not grabbed:
        return # empty video
    dims = frame.shape
    cropped, bounds = cropBlack(frame)
    removed = removeAperture(cropped, bounds)
    interfaces = segmentInterfaces(removed)
    droplets = detectEllipses(interfaces)
    print(droplets)
    imshow(frame, drawEllipses(removed, droplets), interfaces)
    
    stream.release() # close the file
    return frame, removed, interfaces