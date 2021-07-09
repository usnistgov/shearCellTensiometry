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
from sklearn.linear_model import LinearRegression
import re

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from vidMorph import *
from vidCrop import *
from imshow import imshow

plt.rcParams['text.usetex'] = False

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

class Fluid:
    '''holds information about a fluid'''
    
    def __init__(self, s:str):
        '''s is a string that can either be shorthand or longhand for a fluid'''
        self.name = s
        self.base = ''
        self.rheModifier = ''
        self.surfactant = ''
        self.rheWt = 0
        
        if s[0]=='M':
            # mineral oil based fluid
            self.base='mineral oil'
            self.rheModifier = 'fumed silica'
            s = s[1:] # remove M from name
            if s[-1]=='S':
                # contains span
                self.surfactant='Span 20'
                s = s[:-1] # remove S from name
            
        else:
            # Laponite
            self.base = 'water'
            self.rheModifier = 'Laponite RD'
            if s[-1]=='T':
                self.surfactant = 'Tween 80'
                s = s[:-1] # remove t from name
                
        # get wt% from remaining string
        try:
            wt = float(s)
        except:
            logging.warning(s)
            self.rheWt = ''
        else:
            self.rheWt = wt
        self.findRhe()
            
    def findRhe(self) -> None:
        '''find the rheology model from the rheology table'''
        if not os.path.exists(cfg.path.rheTable):
            logging.error(f'No rheology table found: {cfg.path.rheTable}')
            return
        rhe = pd.read_excel(cfg.path.rheTable)
        rhe = rhe.fillna('') 
        entry = rhe[(rhe.base==self.base)&(rhe.rheModifier==self.rheModifier)&(rhe.rheWt==self.rheWt)&(rhe.surfactant==self.surfactant)]
        if len(entry)==0:
            logging.error(f'No rheology fit found for fluid {self.name}')
            return
        if len(entry)>1:
            logging.error(f'Multiple rheology fits found for fluid {self.name}')
        entry = entry.iloc[0]
        self.tau0 = entry['y2_tau0'] # Pa
        self.k = entry['y2_k'] 
        self.n = entry['y2_n']
        self.eta0 = entry['y2_eta0'] # these values are for Pa.s, vs. frequency in rad/s
        return
        
    def visc(self, gdot:float) -> float:
        '''get the viscosity of the fluid in Pa*s at shear rate gdot in rad/s'''
        gdot = (gdot/(2*np.pi)) # convert to Hz
        mu = self.k*(abs(gdot)**(self.n-1)) + self.tau0/(abs(gdot))
        return min(mu, self.eta0)
        
            
class Profile:
    '''stores shear rate step list from file. stores a dataframe and a units dictionary'''
    
    def __init__(self, file:str): 
        with open(file, 'r') as f:
            data = f.read()
            out = ''
            prev=False
            for character in data:
                if character.isalnum() or character=='/':
                    out+=character
                    prev=False
                else:
                    if not prev:
                        out+=','
                        prev=True
            data = out
            spl = [['stdy']+re.split(',',s) for s in re.split('stdy,', data)] # designed for steady steps
            spl = spl[1:] # remove empty first line
            spl = [r[0:4]+r[5:8] for r in spl]
            cols = ['mode', 'gap', 'strain', 'rate', 'freq', 'direction', 'time' ]
            self.units = {'mode':'', 'gap':'um', 'strain':'', 'rate':'rad/s','freq':'','direction':'','time':'s'}
            self.table = pd.DataFrame(spl, columns=cols)

class Test:
    '''stores info about a test for a single material combo and shear profile'''
    
    def __init__(self, folder:str):
        if not os.path.isdir(folder):
            raise NameError('Input to Test must be a folder name')
        self.folder = folder
        bn = os.path.basename(folder)
        spl = re.split('_', bn)
        if not len(spl)==3:
            raise ValueError('Unknown naming format. getFluids assumes format {droplet}_{matrix}_{profile}')
        droplet = spl[0]
        matrix = spl[1]
        profile = spl[2]
    
        self.droplet = Fluid(droplet)
        self.matrix = Fluid(matrix)
        pfile = os.path.join(os.path.dirname(folder), profile+'.mot') # profile should be in parent folder
        if not os.path.exists(pfile):
            raise NameError(f'profile {pfile} not found')
        self.profile = Profile(pfile)
        
        self.videos = [os.path.join(folder, s) for s in os.listdir(folder)] # list of videos
        
    def prnt(self):
        print('Droplet:',self.droplet.__dict__)
        print('Matrix:',self.matrix.__dict__)
        print('Profile:\n', self.profile.table)
        print('Videos:', [os.path.basename(s) for s in self.videos])


            
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

#-----------------------------------

class summaryPlot:
    
    def __init__(self):
        fig, axs = plt.subplots(2,2, figsize=(8,6))
        axs[0,0].set_xlabel(r'$\dot{\gamma}R$ (m/s)')
        axs[0,0].set_ylabel(r'$\frac{D(16\lambda+16)}{\eta_c(19\lambda+16)}$ (m$^2$/(N*s))')
        axs[0,0].set_title('Taylor discrete viscosity')
        axs[1,0].set_xlabel(r'$\dot{\gamma}R\eta_c$ (N/m)')
        axs[1,0].set_ylabel(r'$D$ ()')
        axs[1,0].set_title('Taylor apparent viscosity')
        axs[0,1].set_xlabel(r'$\dot{\gamma}\eta_c$ (N/m^2)')
        axs[0,1].set_ylabel(r'$\sqrt{\frac{r_z-r_0}{r_0^3k}}$ (1/m)')
        axs[0,1].set_title('Greco discrete viscosity')
        axs[1,1].set_xlabel(r'$\dot{\gamma}^2$ (1/$s^2$)')
        axs[1,1].set_ylabel(r'$\frac{r_z-r_0}{r_0^3}$ (1/m$^2$)')
        axs[1,1].set_title('Greco apparent viscosity')
        self.fig = fig
        self.axs = axs
        
    def clean(self):
        self.axs[1,1].legend(bbox_to_anchor=(1.05, 0), loc='lower left')
        self.fig.tight_layout()
        
    def plotSummary(self, d1:pd.DataFrame, label:str='') -> None:
        self.axs[0,0].scatter(d1['gr'],d1['Dte'], label=label, s=4)
        self.axs[1,0].scatter(d1['grem'], d1['D'], label=label, s=4)
        self.axs[0,1].scatter(d1['gem'], d1['srrrk'], label=label, s=4)
        self.axs[1,1].scatter(d1['g2'], d1['rzr0r0'], label=label, s=4)
        
    def plotFit(self, fit:dict) -> None:
        '''plot the fit on the appropriate plot, given mode'''
        
        if fit['mode']==1:
            ax = self.axs[0,0]
        elif fit['mode']==2:
            ax = self.axs[1,0]
        elif fit['mode']==3:
            ax = self.axs[0,1]
        elif fit['mode']==4:
            ax = self.axs[1,1]
        xlist = ax.get_xlim()
        ylist = [fit['slope']*i for i in xlist]
        ax.plot(xlist, ylist, color='black')
        if fit['mode']==4:
            x0 = 0.6
        else:
            x0 = 0.02
        ax.text(x0,0.9, "$\sigma$={:.0f} mJ/m$^2$".format(1000*fit['sigma']), transform=ax.transAxes)
        ax.text(x0,0.8, "$r^2$={:.2f}".format(fit['r2']), transform=ax.transAxes)

#-----------------------------------
    
def grecoK(lam:float) -> float:
    return (42496+191824*lam+261545*lam**2+111245*lam**3)/(60480*(1+lam)**3 )


def linearReg(x:list, y:list, intercept:Union[float, str]='') -> Dict:
    '''Get a linear regression. y=bx+c'''
    y = np.array(y)
    X = np.array(x).reshape((-1,1))
    if type(intercept) is str:
        regr = LinearRegression().fit(X,y)
    else:
        y = y-intercept
        regr = LinearRegression(fit_intercept=True).fit(X,y)
    rsq = regr.score(X,y)
    c = regr.intercept_
    if not type(intercept) is str:
        c = c+intercept
    b = regr.coef_
    b = b[0]
    return {'b':b, 'c':c, 'rsq':rsq}

#-----------------------------------    

class vidInfo:
    '''holds tables of data about the video'''
    
    def __init__(self, vidnum:int, test:Test):
        '''profile is a profile object, defined in fileHandling'''
        self.file = test.videos[vidnum]
        self.profile = test.profile
        self.droplet = test.droplet
        self.matrix = test.matrix
        self.bounds = {}
        self.dropletTab = pd.DataFrame(columns=['frame','time', 'dropNum', 'x', 'y', 'dpos', 'v', 'w', 'l', 'angle'])
        self.dropletTabUnits = {'frame':'','time':'s', 'dropNum':'', 'x':'px', 'y':'px', 'dpos':'px', 'v':'px/s', 'w':'px', 'l':'px', 'angle':'degree'}
        self.relabeledDroplets = []
        self.mppx = 4.487989505128276e-06
        
        
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
        
    def readFrames(self, startFrame:int=0, endFrame:int=100000, reportFreq:int=100) -> None:
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
    
    
    #--------------------------------
    
    
    def consolidateDroplets(self) -> None:
        '''go through the list of droplets generated by readFrames and connect droplets that go offscreen'''
        
        if len(self.dropletTab)==0:
            return
        
        dropNums = self.dropletTab['dropNum'].unique()
        if -1 in dropNums:
            frames = list(self.dropletTab[self.dropletTab['dropNum']==-1]['frame'])
            logging.warning(f'Unlabeled droplets present in frames: {frames}')
            
        self.relabeledDroplets = self.dropletTab.copy()
            
        endpoints = []
        for num in dropNums:
            df = self.dropletTab[self.dropletTab['dropNum']==num]
            firstY = df.iloc[0]['y']>self.bounds['ycc'] # is above midpoint?
            lastY = df.iloc[-1]['y']>self.bounds['ycc'] # is above midpoint?
            if len(df)>1:
                vinit = np.sign(df.iloc[1]['v'])  # velocity at beginning
            else:
                vinit = 0
            vfinal = np.sign(df.iloc[-1]['v']) # velocity at end
            tinit = df.iloc[0]['time']
            tfinal = df.iloc[-1]['time']
            area = df.w.mean()*df.l.mean()*np.pi
            endpoints.append({'dropNum':num, 'firstY':firstY, 'lastY':lastY, 'vinit':vinit, 'vfinal':vfinal, 'tinit':tinit, 'tfinal':tfinal, 'area':area})
        endpoints = pd.DataFrame(endpoints)
        for i, row in endpoints.iterrows():
            if i>0:
                index = endpoints.loc[i].name
                iprev = i-1
                while iprev>=0:
                    try:
                        if endpoints.loc[i,'vinit']==-endpoints.loc[iprev,'vfinal'] and \
                        endpoints.loc[i,'firstY']==endpoints.loc[iprev,'lastY'] and \
                        endpoints.loc[i,'tinit']>endpoints.loc[iprev,'tfinal'] and\
                        abs(endpoints.loc[i, 'area']-endpoints.loc[iprev,'area'])/endpoints.loc[iprev,'area']<0.25:
                            # opposite sign velocities and start/end on same edge and enter after the droplet left and area within 25%: same droplet
                            oldnum = endpoints.loc[i,'dropNum']
                            newnum = endpoints.loc[iprev,'dropNum']
                            self.relabeledDroplets.loc[self.relabeledDroplets['dropNum']==oldnum, 'dropNum']=newnum
                            endpoints.loc[i, 'dropNum']=newnum
                            endpoints.loc[i, 'vinit']=endpoints.loc[iprev]['vinit']
                            endpoints.loc[i, 'firstY']=endpoints.loc[iprev]['firstY']
                            endpoints.drop([iprev], inplace=True)
                            iprev=-1
                        else:
                            iprev=iprev-1
                    except:
                        iprev = iprev-1
                        
        return
    
    #--------------------------------
    
    def plotDroplets(self, relabeled:bool, xstr:str, xmin:float=-1, xmax:float=1000000000, removeOutlier:bool=False, removeZero:bool=True) -> None:
        '''plot the droplets over time. 
        Relabeled true to plot relabeled droplets, false to plot original labels. 
        time true to plot over time. False to plot over frame'''
        
        fig, axs = plt.subplots(4,1, sharex=True)
        axs[0].set_ylabel('width')
        axs[1].set_ylabel('length')
        axs[2].set_ylabel('y')
        axs[3].set_ylabel('velocity')
        
        if relabeled:
            df = self.relabeledDroplets
        else:
            df = self.dropletTab
        if removeZero:
            df = df[(df['v']>0)|(df['v']<0)]
        if xstr=='time':
            axs[3].set_xlabel('time (s)')
        else:
            axs[3].set_xlabel('frame')
        df = df[(df[xstr]>=xmin)&(df[xstr]<=xmax)]
        for n in df.dropNum.unique():
            d1 = df[df['dropNum']==n]
            if removeOutlier:
                d1 = removeOutliers(d1, 'w')
                d1 = removeOutliers(d1, 'v')
            axs[0].scatter(d1[xstr],d1['w'], label=n, s=2)
            axs[1].scatter(d1[xstr], d1['l'], label=n, s=2)
            axs[2].scatter(d1[xstr], d1['y'], label=n, s=2)
            axs[3].scatter(d1[xstr], d1['v'], label=n, s=2)
        axs[3].legend(bbox_to_anchor=(1.05, 0), loc='lower left')
    
    
        
    #--------------------------------------
    
    def baselineDroplets(self) -> None:
        '''determine baseline undeformed droplet dimensions'''
        if len(self.relabeledDroplets)==0:
            self.consolidateDroplets()
            if len(self.relabeledDroplets)==0:
                logging.info('baselineDroplets aborted. No data in relabeledDroplets')
                return
        
        statics = self.relabeledDroplets[(self.relabeledDroplets.dpos<1)&(self.relabeledDroplets.dpos>-1)]
        baselines = []
        for dropNum in self.relabeledDroplets.dropNum.unique():
            # find average within the zero velocity regions
            stati = statics[statics.dropNum==dropNum]
            if len(stati)>0:
                wmean = stati.w.mean()
                lmean = stati.l.mean()
                vol = ((wmean/2)**2)*(lmean/2) # assume that the droplet has depth w
                req = (vol)**(1/3) # equivalent radius of droplet volume
            else:
                wmean=-1
                lmean=-1
                req=-1
            
            # alt method: find smallest difference:
            stati = self.relabeledDroplets[self.relabeledDroplets.dropNum==dropNum]
            diff = [row['l']-row['w'] for i,row in stati.iterrows()]
            row = stati.iloc[diff.index(min(diff))]
            w2 = row['w']
            l2 = row['l']
            req2 = ((w2/2)**2*(l2/2))**(1/3)
            baselines.append({'dropNum':dropNum, 'w':wmean, 'l':lmean, 'r0':req, 'w2':w2, 'l2':l2, 'r02':req2})
        self.baselines = pd.DataFrame(baselines)
        self.baselinesUnits = {'dropNum':'', 'w':'px', 'l':'px', 'r0':'px', 'w2':'px', 'l2':'px', 'r02':'px'}
        return
    
    def splitTimes(self) -> None:
        '''split the relabeled droplets into moves, using a Profile object from fileHandling'''
        profile = self.profile
        
        if len(self.relabeledDroplets)==0:
            self.consolidateDroplets()
            if len(self.relabeledDroplets)==0:
                logging.info('splitTimes aborted. No data in relabeledDroplets')
                return
            
        df = self.relabeledDroplets
        startTime = df[df.dpos>1].time.min() # first time where the droplet is moving more than 1 px/frame
        currTime = startTime
        moveTimes = []  # list of times where the new step starts
        
        progTimes = profile.table.time.astype(float)
        totalVidTime = df.time.max()-startTime  # total length of video
        totalProgTime = progTimes.sum() # total programmed length
        scale = totalVidTime/totalProgTime # ratio that time is compressed by in actual video
        
        for i,row in profile.table.iterrows():
            dt = float(row['time'])*scale
            dfrange = df[(df.time>=currTime)]
            # filter by velocity
            if row['direction']=='cw':
                # velocity should be positive
                dfrange = dfrange[(dfrange.v>10)]
            else:
                dfrange = dfrange[(dfrange.v<-10)]
            currTime = dfrange.time.min()
            
            # filter by time
            dfrange = dfrange[(dfrange.time>=currTime)&(dfrange.time<=currTime+dt)]

            # get stats on this range
            vave = dfrange.v.mean()
            vmed = dfrange.v.median()
            firstTime = dfrange.time.min()
            finalTime = dfrange.time.max()
            currTime = finalTime # update the time to search from to the final time
            
            rowkeep = {'gap':row['gap'], 'rate':row['rate'], 'direction':row['direction'], 'tprog':row['time']} # info from profile table
            dropStat = {'t0':firstTime, 'tf':finalTime, 'vave':vave, 'vmed':vmed, 'vave/rate':abs(vave/float(row['rate'])),'vmed/rate':abs(vmed/float(row['rate']))} # info from relabeledDroplets
            keep = {**rowkeep, **dropStat} 
            moveTimes.append(keep)
            
        self.moveTimes = pd.DataFrame(moveTimes)
        self.moveTimesUnits = {'gap':self.profile.units['gap'], 'rate':self.profile.units['rate'], 'direction':'', 'tprog':self.profile.units['time'], 't0':'s', 'tf':'s', 'vave':'px/s', 'vmed':'px/s', 'vave/rate':'px/rad', 'vmed/rate':'px/rad'}
            
        return 
    
    
    #--------------------------------
    
    def summarizeDroplet(self, dropNum:int) -> pd.DataFrame:
        '''summarize measurements of the given droplet number'''
        df = self.relabeledDroplets[self.relabeledDroplets.dropNum==dropNum]
        if len(df)==0:
            return
        baseline = self.baselines[self.baselines.dropNum==dropNum]
        if len(baseline)==0:
            logging.info(f'No baseline radius for droplet {dropNum}')
            return
        baseline = baseline.iloc[0]
        r0 = baseline['r0']*self.mppx # convert this to meters
        measurements = []
        for i,row in self.moveTimes.iterrows():
            t0 = row['t0']
            tf = row['tf']
            df0 = df[(df.time>=t0)&(df.time<=tf)] # points within this time range
            wave = df0.w.mean()*self.mppx # convert this to meters
            rz = wave/2
            lave = df0.l.mean()*self.mppx
            
            if wave>0 and lave>0:

                gdot = float(row['rate']) # shear rate in rad/s
                gdothz = gdot/(2*np.pi) # shear rate in hz
                etad = self.droplet.visc(gdot) # viscosity of droplet (Pa*s)
                etam = self.matrix.visc(gdot) # viscosity of matrix (Pa*s)
                lam = etad/etam # viscosity ratio
                k = grecoK(lam) # Greco eq, unitless

                # Taylor method apparent viscosity: slope = 1/sigma*(19lambda+16)/(16lambda+16)
                gdotRetam = gdothz*r0*etam # Pa*m
                D = (lave-wave)/(lave+wave)

                # Taylor method discrete viscosities: slope = 1/sigma
                gdotR = gdothz*r0 # (m/s)
                Dtayloretam = D*(16*lam+16)/((19*lam+16)*etam) # 1/(Pa*s)

                # Greco method apparent viscosity: slope = k(etam/sigma)^2
                g2 = gdothz**2 # (1/s^2)
                rzr0r0 = (rz-r0)/r0**3

                # Greco method discrete viscosities: slope = 1/sigma
                gdotetam = gdot*etam # Pa*m/s
                srrrk = np.sqrt(-rzr0r0/k)

                measurements.append({'dropNum':dropNum, 'gdot':gdot, 'gdotHz':gdothz, 'etad':etad, 'etam':etam, 'lam':lam, 'k':k, 'w':wave, 'l':lave, 'r0':r0, 'grem':gdotRetam, 'D':D, 'gr':gdotR, 'Dte':Dtayloretam, 'g2':g2, 'rzr0r0':rzr0r0, 'gem':gdotetam, 'srrrk':srrrk})
            
        return pd.DataFrame(measurements)
            
    
    def summarizeDroplets(self) -> None:
        '''take final measurements for all droplets'''
        self.summary = pd.DataFrame()
        
        for dropNum in self.relabeledDroplets.dropNum.unique():
            s = self.summarizeDroplet(dropNum)
            self.summary = pd.concat([self.summary, s], ignore_index=True) # add to the complete list
        self.summaryUnits = {'dropNum':'', 'gdot':'rad/s', 'gdotHz':'1/s', 'etad':'Pa.s', 'etam':'Pa.s', 'lam':'', 'k':'', 'w':'m', 'l':'m', 'r0':'m', 'grem':'N/m', 'D':'', 'gr':'m/s', 'Dte':'m^2/(N*s)', 'g2':'1/s^2', 'rzr0r0':'1/m^2', 'gem':'N/m^2', 'srrrk':'1/m'}
        return
    
    def getFit(self, mode:int) -> dict:
        '''fit the summary data using one of the 4 methods'''
        if mode==1:
            # method 1: Taylor discrete viscosity
            summ = self.summary
            fit = linearReg(summ.gr, summ.Dte, intercept=0)
            sigma = 1/fit['b'] # N/m
        elif mode==2:
            # method 2: Taylor apparent viscosity
            lam = self.summary.lam.mean() # average lambda
            fit = linearReg(self.summary.grem, self.summary.D, intercept=0)
            sigma = 1/fit['b']*(19*lam+16)/(16*lam+16) # N/m
        elif mode==3:
            # method 3: Greco discrete viscosity
            fit = linearReg(self.summary.gem, self.summary.srrrk, intercept=0)
            sigma = 1/fit['b'] # N/m
        elif mode==4:
            # method 4: Greco apparent viscosity
            lam = self.summary.lam.mean() # average lambda
            etac = self.summary.etam.mean() # average matrix viscosity
            k = grecoK(lam)
            fit = linearReg(self.summary.g2, self.summary.rzr0r0, intercept=0)
            sigma = etac/np.sqrt(-fit['b']/k) # N/m  
                
        return {'mode':mode, 'sigma':sigma, 'slope':fit['b'], 'r2':fit['rsq']}
    
    def getSigma(self, plot:bool=False) -> None:
        '''calculate sigma 4 ways'''
        fits = dict([[i+1,self.getFit(i+1)] for i in range(4)])
        self.fits = fits
        if plot:
            for i in fits:
                self.summaryPlot.plotFit(fits[i])
        return fits
    
    def exportSigma(self, folder:str, name:str) -> None:
        fn = os.path.join(folder, 'sigma_'+name+'.csv')
        with open(fn, 'w') as f:
            for key in self.fits.keys():
                sigma = self.fits[key]
                for key1 in sigma.keys():
                    f.write("%s_%s,"%(key1,key))
            f.write("\n")
            units=['m/N', 'm/N', 'm/N', 's^2/m^2']
            for i,key in enumerate(self.fits.keys()):
                f.write(',N/m,%s,,'%(units[i]))
            f.write("\n")
            for key in self.fits.keys():
                sigma = self.fits[key]
                for key1 in sigma.keys():
                    f.write("%s,"%(sigma[key1]))
                    
    def exportSummary(self, folder:str, name:str) -> None:
        col = pd.MultiIndex.from_tuples([(k,v) for k, v in self.summaryUnits.items()])
        data = np.array(self.summary)
        df = pd.DataFrame(data, columns=col)
        df.to_csv(os.path.join(folder, 'summary_'+name+'.csv'))
        
    def exportDroplets(self, folder:str, name:str) -> None:
        col = pd.MultiIndex.from_tuples([(k,v) for k, v in self.dropletTabUnits.items()])
        data = np.array(self.relabeledDroplets)
        df = pd.DataFrame(data, columns=col)
        df.to_csv(os.path.join(folder, 'droplets_'+name+'.csv'))
    
    def plotSummaries(self, splitDroplets:bool=True, label:str='') -> None:
        '''plot summary data'''
        sp = summaryPlot()
        df = self.summary
        if splitDroplets:
            for n in df.dropNum.unique():
                d1 = df[df['dropNum']==n]
                sp.plotSummary(d1, label=label+"_"+str(int(n)))
        else:
            sp.plotSummary(df, label=label)
        self.summaryPlot = sp
        self.getSigma(plot=True) # calculate sigmas
        self.summaryPlot.clean()
        
        
    #----------------------------------
    
    def exportAll(self) -> None:
        '''export all tables'''
        folder = cfg.path.export
        times = re.split('_', os.path.splitext(os.path.basename(self.file))[0])
        name = self.droplet.name+'_'+self.matrix.name+'_'+times[-2]+'_'+times[-1]
        self.exportSummary(folder, name)
        self.exportDroplets(folder, name)
        self.exportSigma(folder, name)
        
        
    #---------------------------------
    
    def analyze(self, plot:bool=True) -> None:
        '''analyze the entire video. If we already started this, pick up where we left off.'''
        if len(self.dropletTab)==0:
            startFrame = 0
        else:
            startFrame = max(self.dropletTab.frame)
        self.readFrames(startFrame=startFrame)
        logging.info('Consolidating droplets')
        self.consolidateDroplets()
        logging.info('Determining baseline')
        self.baselineDroplets()
        logging.info('Splitting times')
        self.splitTimes()
        logging.info('Summarizing droplets')
        self.summarizeDroplets()
        logging.info('Getting sigma')
        self.getSigma()
        self.plotSummaries(True,label=self.droplet.name+'_'+self.matrix.name)