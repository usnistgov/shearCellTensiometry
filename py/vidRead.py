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



            
############ FRAME READING TOOLS

def closest_node(node:Tuple, nodes:Tuple) -> int:
    '''Find closest point to list of points. https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points'''
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def readSpecificFrame(file:str, time:float=-1, frameNum:int=-1) -> np.array:
    '''read a specific frame from the video. https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture'''
    stream = cv.VideoCapture(file) # open the file
    if time>0:
        fps = stream.get(cv.CAP_PROP_FPS)
        frameNum = round(time*fps)
    if frameNum<0:
        return []
    stream.set(1, max(frameNum-1, 0))

    #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
    ret, frame = stream.read()
    
    stream.release() # close the file
    return frame

def removeOutliers(dfi:pd.DataFrame, col:str, low:float=0.05, high:float=0.95) -> pd.DataFrame:
    '''https://nextjournal.com/schmudde/how-to-remove-outliers-in-data'''
    df = dfi.copy()
    y = df[col]
    removed_outliers = y.between(y.quantile(low), y.quantile(high))
    index_names = df[~removed_outliers].index
    df.drop(index_names, inplace=True)
    return df
    
    

class vidInfo:
    '''holds tables of data about the video'''
    
    def __init__(self, file:str):
        self.file = file
        self.bounds = {}
        self.dropletTab = pd.DataFrame(columns=['frame','time', 'dropNum', 'x', 'y', 'dpos', 'v', 'w', 'l', 'angle'])
        self.dropletTabUnits = {'frame':'','time':'s', 'dropNum':'', 'x':'px', 'y':'px', 'dpos':'px', 'v':'px/s', 'w':'px', 'l':'px', 'angle':'degree'}
        
    def detectDropletOneFrame(self, time:float=-1, frameNum:int=-1, diag:bool=False) -> List[tuple]:
        '''detect droplets for just one frame'''
        frame = readSpecificFrame(self.file, time=time, frameNum=frameNum)
        droplets = self.getDroplets(frame, diag=diag)
        return droplets
        
    def getDroplets(self, frame:np.array, diag:bool=False) -> int:
        '''get droplets from a frame'''
        # crop to aperture
        if len(self.bounds)==0:
            cropped, bounds = cropBlack(frame)
            self.bounds = bounds
        else:
            cropped, bounds = cropBlack(frame, bnds=self.bounds)
            
        removed = removeAperture(cropped, bounds)          # grey out everything outside of aperture
        interfaces = segmentInterfaces(removed)            # get B&W blobs of fluid interfaces
        interfaces2 = eliminateTouching(interfaces, bounds) # get rid of any blobs touching the aperture
        droplets = detectEllipses(interfaces2, diag=False)              # measure ellipses
        if diag:
            ellipse1 = drawEllipses(removed, droplets, diag=False)
            imshow(ellipse1, interfaces, interfaces2)
        return droplets
    
    #---------------------------------
    
    def dropletChange(self, oldrow:pd.Series, newrow:pd.Series, fps:int) -> dict:
        '''determine change in droplet'''
        x = newrow['x'] # find position of new droplet
        y = newrow['y']
        # determine change in position, speed
        xprev = oldrow['x']
        yprev = oldrow['y']
        dx = x-xprev
        dy = y-yprev
        dd = np.sqrt(dx**2+dy**2)*np.sign(dy) # positive dd means droplet is traveling in positive y
        v = dd*fps
        return {'dpos':dd, 'v':v}
    
    #---------------------------------
    
    def checkStatic(self, newDroplets:pd.DataFrame, fps:int) -> Tuple[bool, pd.DataFrame]:
        if len(newDroplets)==0 or len(self.prevDroplets)==0:
            return False, newDroplets
        if not len(newDroplets)==len(self.prevDroplets):
            # different number of droplets: stage is not static
            return False, newDroplets
        changes= [self.dropletChange(self.prevDroplets.loc[i], newDroplets.loc[i], fps) for i in range(len(newDroplets))]
        distances = [i['dpos'] for i in changes]
        if np.mean(distances)<5 or max(distances)-min(distances)<20:
            # change is less than 5 px or all of the distances are within 20px of each other. copy all droplet numbers directly
            for i in range(len(newDroplets)):
                newDroplets.loc[i,'dropNum'] = self.prevDroplets.loc[i,'dropNum']
                newDroplets.loc[i,'dpos'] = changes[i]['dpos']
                newDroplets.loc[i,'v'] =changes[i]['v']
            return True, newDroplets
        else:
            return False, newDroplets
                
    
    def findCritDD(self, newDroplets:pd.DataFrame, fps:int) -> float:
        '''find the critical distance above which the new droplet must move'''
        critdd = 0
        dave = self.prevDroplets['dpos'].mean() # average velocity of previous droplets
        if abs(dave)<5:
            # change is too small. don't filter
            critdd=0
        elif dave<0 or dave>0:
            critdd = dave # if the previous droplets were known to be moving, the new move must be at least half of the previous move
        elif len(self.bounds)>0:
            # previous droplet was not linked to another droplet, empty velocity. Determine if it just entered the frame
            prevYmaxI = self.prevDroplets['y'].idxmax()
            prevYmax = self.prevDroplets.loc[prevYmaxI, 'y']
            if prevYmax<self.bounds['ycc']:
                newYMaxI = newDroplets['y'].idxmax()
                newYmax = newDroplets.loc[newYMaxI, 'y']
                if prevYmax<newYmax:
                    # max droplet had just entered frame from y=0 and new droplets are at higher y
                    critdd = self.dropletChange(self.prevDroplets.loc[prevYmaxI], newDroplets.loc[newYMaxI], fps)['dpos']
            else:
                prevYminI = self.prevDroplets['y'].idxmin()
                prevYmin = self.prevDroplets.loc[prevYminI, 'y']
                if prevYmin>self.bounds['ycc']:
                    newYMinI = newDroplets['y'].idxmin()
                    newYmin = newDroplets.loc[newYMinI, 'y']
                    if prevYmax>newYmax:
                        # min droplet had just entered frame from y=max and new droplets are at lower y
                        critdd = self.dropletChange(self.prevDroplets.loc[prevYmaxI], newDroplets.loc[newYMaxI], fps)['dpos']
        critdd = critdd*0.25
        return critdd
    
    def closestDroplet(self, i:int, newDroplets:pd.DataFrame, fps:int, critdd:float, excludeAssigned:bool=False) -> None:
        '''find the closest droplet to droplet i from the previous frame. '''
        row = self.prevDroplets.iloc[i]
        if critdd<0:
            # select only droplets with dd < critdd
            dds = [self.dropletChange(row, newrow, fps)['dpos']<critdd for i,newrow in newDroplets.iterrows()]
            nd = newDroplets[dds]
        elif critdd>0:
            # select only droplets with dd > critdd
            dds = [self.dropletChange(row, newrow, fps)['dpos']>critdd for i,newrow in newDroplets.iterrows()]
            nd = newDroplets[dds]
        else:
            # select all droplets
            nd = newDroplets
        if excludeAssigned:
            # exclude droplets that have already been assigned
            nd = newDroplets[newDroplets['dropNum']<0]
        
        retry=True
        while retry:
            if len(nd)==0:
                return
            ival = closest_node(tuple(row[['x','y']]), nd[['x','y']]) # index of closest droplet
            val = nd.iloc[ival].name # get index value for actual newDroplets
            newDropletRow = newDroplets.loc[val]
            change = self.dropletChange(row, newDropletRow, fps) # find the velocity, change in position
            if newDroplets.loc[val, 'dropNum']>=0:
                # droplet is already assigned. 
                if abs(newDroplets.loc[val, 'dpos']) < abs(change['dpos']):
                    # existing assignment is better fit
                    nd=nd.drop([val])
                else:
                    # this assignment is better fit
                    retry=False
                    prevnum = newDroplets.loc[val, 'dropNum']
                    redodroptab = self.prevDroplets[self.prevDroplets['dropNum']==prevnum]
                    redoval = redodroptab.iloc[0].name # index
                    newDroplets.loc[val, 'dropNum'] = row['dropNum'] # set dropNum
                    self.closestDroplet(redoval, newDroplets, fps, critdd, excludeAssigned=True)
            else:
                retry=False
        newDroplets.loc[val, 'dropNum'] = row['dropNum'] # set dropNum
        newDroplets.loc[val, 'dpos']=change['dpos']
        newDroplets.loc[val, 'v']=change['v']
        
    #--------------------------------------
    
    def assignFromPrevious(self, newDroplets:pd.DataFrame, fps:int) -> pd.DataFrame:
        '''assign new droplet numbers from previous droplet numbers'''
        if len(newDroplets)==0 or len(self.prevDroplets)==0:
            return newDroplets
        
        critdd = self.findCritDD(newDroplets, fps)
        # for each droplet in the previous timestep, find the closest droplet and reassign
        for i in self.prevDroplets.index:
            self.closestDroplet(i, newDroplets, fps, critdd)
        return newDroplets

    
    def assignNewDropletNumbers(self, newDroplets:pd.DataFrame) -> pd.DataFrame:
        '''assign new droplet numbers to unmatched droplets'''
        if len(self.dropletTab['dropNum'])>0:
            dropCounter = max(self.dropletTab['dropNum'])+1
        else:
            dropCounter = 0
        for i,row in newDroplets.iterrows():
            if row['dropNum']<0:
                newDroplets.loc[i,'dropNum'] = dropCounter
                newDroplets.loc[i, 'dpos']=0
                newDroplets.loc[i, 'v']=0
                dropCounter+=1
        return newDroplets
    

    
    def assignDroplets(self, newDroplets:pd.DataFrame, fps:int) -> pd.DataFrame:
        '''assign droplet numbers'''
        # check if static
        static, newDroplets = self.checkStatic(newDroplets, fps)
        if static:
            return newDroplets
        # match droplets to previous droplets
        self.assignFromPrevious(newDroplets, fps)
        # assign new numbers to all droplets that weren't matched to an existing droplet
        self.assignNewDropletNumbers(newDroplets)
        return newDroplets
    
    #-----------------------------------
    
    def measureDroplets(self, droplets:List[tuple], time:float, frame:int, fps:int) -> None:
        '''measure droplets and add them to the droplet list'''
        newDroplets = []
        for d in droplets:
            pos = d[0]
            dims = d[1]
            angle = d[2]
            newDroplets.append({'frame':frame, 'time':time,'dropNum':-1,'x':pos[0],'y':pos[1],'dpos':0,'v':0,'w':dims[0],'l':dims[1],'angle':angle})
        
        df = pd.DataFrame(newDroplets)
        df = self.assignDroplets(df, fps)
        return df
        
    def streamInfo(self, stream, fps) -> Tuple:
        time = stream.get(cv.CAP_PROP_POS_MSEC)/1000
        frame = stream.get(cv.CAP_PROP_POS_FRAMES)
        if fps==0:
            fps=stream.get(cv.CAP_PROP_FPS)
        return time, frame, fps
        
    def getEllipses(self, stream:cv.VideoCapture, fps:float=0) -> int:
        '''get info about the ellipses. Returns 1 when video is done. Returns 0 to continue grabbing.'''
        grabbed, frame = stream.read() # read first frame
        if not grabbed:
            return 1 # video is done
        
        droplets = self.getDroplets(frame)
        time, frame, fps = self.streamInfo(stream, fps)
        self.prevDroplets = self.measureDroplets(droplets, time, frame, fps) # save this timestep 
        self.dropletTab = pd.concat([self.dropletTab, self.prevDroplets], ignore_index=True) # add to the complete list

        return 0
        
    def readFrames(self, startFrame:int=0, endFrame:int=100000, reportFreq:int=1000) -> None:
        '''iterate through frames'''
        file = self.file
        stream = cv.VideoCapture(file) # open the file
        fps = stream.get(cv.CAP_PROP_FPS)
        self.prevDroplets = pd.DataFrame(columns=['frame','time', 'dropNum', 'x', 'y', 'dpos', 'v', 'w', 'l', 'angle'])
        self.dropletTab = self.dropletTab[(self.dropletTab['frame']<startFrame)|(self.dropletTab['frame']>endFrame)]
        stream.set(1, startFrame-1) # set to initial frame

        ret = 0
        frames = startFrame
        while ret==0 and frames<=endFrame:
            if frames%reportFreq==0:
                logging.info(f'Analyzing frame {frames+1}')
            try:
                ret = self.getEllipses(stream, fps=fps)
                frames+=1
            except Exception as e:
                logging.error(f'Error on frame {frames+1}')
                traceback.print_exc(limit=None, file=None, chain=True)
                return
        stream.release() # close the file
        return 