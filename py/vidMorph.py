#!/usr/bin/env python
'''Morphological operations'''

# external packages
import cv2 as cv
import imutils
import numpy as np 
import os
import sys
import logging
from typing import List, Dict, Tuple, Union, Any, TextIO

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
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


######### MORPHOLOGICAL OPERATIONS

def morph(img:np.array, width:int, func:str, iterations:int=1, shape:bool=cv.MORPH_RECT, aspect:float=1, **kwargs) -> np.array:
    '''erode, dilate, open, or close. func should be erode, dilate, open, or close. aspect is aspect ratio of the kernel, height/width'''
    if not shape in [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS]:
        raise NameError('Structuring element must be rect, ellipse, or cross')
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(width, int(width*aspect)))
    if func=='erode':
        return cv.erode(img, kernel, iterations = iterations)
    elif func=='dilate':
        return cv.dilate(img, kernel, iterations = iterations)
    elif func=='open':
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif func=='close':
        return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    else:
        raise NameError('func must be erode, dilate, open, or close')

def erode(img:np.array, size:int, **kwargs) -> np.array:
    '''dilate an image, given a kernel size. https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html'''
    return morph(img, size, 'erode', **kwargs)
    
    
def dilate(img:np.array, size:int, **kwargs) -> np.array:
    '''dilate an image, given a kernel size. https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html'''
    return morph(img, size, 'dilate', **kwargs)

def openMorph(img:np.array, size:int, **kwargs) -> np.array:
    '''open the image (erode then dilate)'''
    return morph(img, size, 'open', **kwargs)

def closeMorph(img:np.array, size:int, **kwargs) -> np.array:
    '''close the image (dilate then erode)'''
    return morph(img, size, 'close', **kwargs)



########### SEGMENTATION

def componentCentroid(img:np.array, label:int) -> List[int]:
    '''identify the centroid of a labeled component. Returns label, x, y, size'''
    mask = np.where(img == label)
    x = int(np.mean(mask[0]))
    y = int(np.mean(mask[1]))
    return [label,x,y,len(mask[0])]

def componentCentroids(img:np.array) -> np.array:
    '''given a labeled image, get a list of all of the centroids of the labeled components'''
    labels = list(np.unique(img))
    centroids = [componentCentroid(img, l) for l in labels]
    return centroids  


def fillComponents(thresh:np.array)->np.array:
    '''fill the connected components in the thresholded image. https://www.programcreek.com/python/example/89425/cv2.floodFill'''
    im_flood_fill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
    img_out = thresh | im_flood_fill_inv
    return img_out

def segmentInterfaces(img:np.array) -> np.array:
    '''extract just the ink-support interfaces, which are dark'''
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    _, interfaces = cv.threshold(gray,100,255,cv.THRESH_BINARY_INV)
    interfaces = openMorph(interfaces, 5)
    filled = fillComponents(interfaces)
    return filled

def detectCircles(interfaces:np.array) -> np.array:
    '''detect circles from a filled b+w image of the fluid interfaces'''
    dp = 3
    circles = []
    while len(circles)==0 or len(circles)>20:
        circles = cv.HoughCircles(interfaces, cv.HOUGH_GRADIENT, dp, 50)
    return circles

def drawCircles(img:np.array, circles:np.array) -> None:
    removed2 = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(removed2, center, 1, (0, 100, 100), 5)
            # circle outline
            radius = i[2]
            cv.circle(removed2, center, radius, (255, 0, 255), 3)
    return removed2


def detectEllipses(interfaces:np.array, diag:bool=False) -> list:
    '''find position, dimensions, and angle of ellipses fit to interfaces. diag=true to draw result'''
    edge = imutils.auto_canny(interfaces) # just get edges of droplets
    dilated = dilate(edge, 4) # thicken edges
    _, contours, hierarchy = cv.findContours(interfaces, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # interfaces
    drawing = np.zeros((interfaces.shape[0], interfaces.shape[1]), dtype=np.uint8) # empty b&w drawing
    ellipseOut = []
    for contour in contours:
        color = (256, 256, 256) 
        dr = drawing.copy() # create a copy of the drawing
        ellipse = cv.fitEllipse(contour) # fit the ellipse to the contour for that droplet
        cv.ellipse(dr, ellipse, color, 2) # draw the droplet on the drawing w/ radius 2
        combined = cv.bitwise_and(dilated, dilated, mask=dr)
        overlapPx = cv.sumElems(combined)[0] # number of pixels in common between ellipse and original edges
        ellipsePx = cv.sumElems(dr)[0] # number of pixels in ellipse
        if overlapPx > 0.7*ellipsePx:
            ellipseOut.append(ellipse)
    if diag:
        annotated = drawEllipses(dilated, ellipseOut, diag=True)
    return ellipseOut

def drawEllipses(img:np.array, ellipses:List, diag:bool=False) -> None:
    if len(img.shape)==2:
        annotated = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        annotated = img
    for ellipse in ellipses:
        cv.ellipse(annotated, ellipse, (0,0,200), 3)
    if diag:
        imshow(annotated)
    return annotated
    
    
    
