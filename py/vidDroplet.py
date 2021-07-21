#!/usr/bin/env python
'''Morphological operations'''

# external packages
import pandas as pd
import numpy as np
import os
import sys
import logging
import cv2 as cv
from sklearn.linear_model import LinearRegression

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from vidMorph import *
from vidCrop import *
from config import cfg

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

def streamInfo(stream) -> Tuple:
    time = stream.get(cv.CAP_PROP_POS_MSEC)/1000
    frame = stream.get(cv.CAP_PROP_POS_FRAMES)
    return time, frame


def dropletChange(oldrow:pd.Series, newrow:pd.Series, fps:int) -> dict:
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

def closest_node(node:Tuple, nodes:Tuple) -> int:
    '''Find closest point to list of points. https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points'''
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def endMatch(drop:pd.Series, dropprev:pd.Series) -> bool:
    '''do these endpoints match?'''
    if drop['tinit']<dropprev['tfinal']:
        # first time of new leg is before last time of previous leg. endpoints don't match
        return False
    areaDiff = abs(drop['area']-dropprev['area'])/dropprev['area']
    if (drop['vinit']*dropprev['vfinal']<=0):
        # going opposite directions
        if drop['firstY']==dropprev['lastY'] and areaDiff<0.25:
            # new drop starts and prev stop ends on same side of the window and areas are similar
            return True
    else:
        # going same direction
        dy = drop['y0']-dropprev['yf']
        if np.sign(dy)*dropprev['vfinal']>=0:
            # change in y is in same direction as final velocity
            dy = abs(dy)
            dx = abs(dropprev['xf']-drop['x0'])
            dframe = drop['frame0']-dropprev['framef']
            if dy<30 and dx<30 and dframe<10:
                # nearly same position and area: missed a couple of frames
                return True
    # not met either criteria: return false
    return False

        


#---------------


class dropletTracker:
    '''the purpose of this object is to store tables, where each row represents a droplet in a single frame'''
    
    def __init__(self, test, vidnum:int): 
        '''test is a Test object, defined in fileHandling'''
        self.file = test.videos[vidnum]
        self.bounds = {}
        self.emptyDropTab = pd.DataFrame(columns=['frame','time', 'dropNum', 'x', 'y', 'dpos', 'v', 'w', 'l', 'angle'])
        self.dropletTab = self.emptyDropTab.copy()
        self.prevDroplets = self.emptyDropTab.copy()
        self.dropletTabUnits = {'frame':'','time':'s', 'dropNum':'', 'x':'px', 'y':'px', 'dpos':'px', 'v':'px/s', 'w':'px', 'l':'px', 'angle':'degree'}
        stream = cv.VideoCapture(self.file)
        self.fps = stream.get(cv.CAP_PROP_FPS)
        self.totalFrames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))
        stream.release()
        self.finalFrame = 0

#---------------

    def checkStatic(self, newDroplets:pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        if len(newDroplets)==0 or len(self.prevDroplets)==0:
            return False, newDroplets
        if not len(newDroplets)==len(self.prevDroplets):
            # different number of droplets: stage is not static
            return False, newDroplets
        changes= [dropletChange(self.prevDroplets.loc[i], newDroplets.loc[i], self.fps) for i in range(len(newDroplets))]
        distances = [i['dpos'] for i in changes]
        if np.mean(distances)<5 or max(distances)-min(distances)<20:
            # change is less than 5 px or all of the distances are within 20px of each other. copy all droplet numbers directly
            for i in range(len(newDroplets)):
                newDroplets.loc[i,'dropNum'] = self.prevDroplets.loc[i,'dropNum']
                newDroplets.loc[i,'dpos'] = changes[i]['dpos']
                newDroplets.loc[i,'v'] = changes[i]['v']
            return True, newDroplets
        else:
            return False, newDroplets

    #---------------

    def findCritDD(self, newDroplets:pd.DataFrame) -> float:
        '''find the critical distance above which the new droplet must move'''
        critdd = 0
        dave = self.prevDroplets['dpos'].mean() # average velocity of previous droplets
        if abs(dave)<5:
            # change is too small. don't filter
            critdd=0
        elif dave<0 or dave>0:
            critdd = dave 
            # if the previous droplets were known to be moving,
            # the new move must be at least half of the previous move
        elif len(self.bounds)>0:
            # previous droplet was not linked to another droplet,
            # empty velocity. Determine if it just entered the frame
            prevYmaxI = self.prevDroplets['y'].idxmax()
            prevYmax = self.prevDroplets.loc[prevYmaxI, 'y']
            if prevYmax<self.bounds['ycc']:
                newYMaxI = newDroplets['y'].idxmax()
                newYmax = newDroplets.loc[newYMaxI, 'y']
                if prevYmax<newYmax:
                    # max droplet had just entered frame from y=0 
                    # and new droplets are at higher y
                    critdd = dropletChange(self.prevDroplets.loc[prevYmaxI], newDroplets.loc[newYMaxI], self.fps)['dpos']
            else:
                prevYminI = self.prevDroplets['y'].idxmin()
                prevYmin = self.prevDroplets.loc[prevYminI, 'y']
                if prevYmin>self.bounds['ycc']:
                    newYMinI = newDroplets['y'].idxmin()
                    newYmin = newDroplets.loc[newYMinI, 'y']
                    if prevYmax>newYmax:
                        # min droplet had just entered frame from y=max
                        # and new droplets are at lower y
                        critdd = dropletChange(self.prevDroplets.loc[prevYmaxI], newDroplets.loc[newYMaxI], self.fps)['dpos']
        critdd = critdd*0.25
        return critdd

    #---------------

    def closestDroplet(self, i:int, newDroplets:pd.DataFrame, critdd:float, excludeAssigned:bool=False) -> None:
        '''find the closest droplet to droplet i from the previous frame. '''
        row = self.prevDroplets.iloc[i]
        if critdd<0:
            # select only droplets with dd < critdd
            dds = [dropletChange(row, newrow, self.fps)['dpos']<critdd for i,newrow in newDroplets.iterrows()]
            nd = newDroplets[dds]
        elif critdd>0:
            # select only droplets with dd > critdd
            dds = [dropletChange(row, newrow, self.fps)['dpos']>critdd for i,newrow in newDroplets.iterrows()]
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
            change = dropletChange(row, newDropletRow, self.fps) # find the velocity, change in position
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
                    self.closestDroplet(redoval, newDroplets, critdd, excludeAssigned=True)
            else:
                retry=False
        newDroplets.loc[val, 'dropNum'] = row['dropNum'] # set dropNum
        newDroplets.loc[val, 'dpos']=change['dpos']
        newDroplets.loc[val, 'v']=change['v']

    #--

    def assignFromPrevious(self, newDroplets:pd.DataFrame) -> pd.DataFrame:
        '''assign new droplet numbers from previous droplet numbers'''
        if len(newDroplets)==0 or len(self.prevDroplets)==0:
            return newDroplets

        critdd = self.findCritDD(newDroplets)
        # for each droplet in the previous timestep, find the closest droplet and reassign
        for i in self.prevDroplets.index:
            self.closestDroplet(i, newDroplets, critdd)
        return newDroplets

    #---------------

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

    #---------------

    def assignDroplets(self, newDroplets:pd.DataFrame) -> pd.DataFrame:
        '''assign droplet numbers'''
        # check if static
        static, newDroplets = self.checkStatic(newDroplets)
        if static:
            return newDroplets
        # match droplets to previous droplets
        newDroplets = self.assignFromPrevious(newDroplets)
        # assign new numbers to all droplets that weren't matched to an existing droplet
        newDroplets = self.assignNewDropletNumbers(newDroplets)
        return newDroplets
    
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
    
    def measureDroplets(self, droplets:List[tuple], time:float, frame:int) -> None:
        '''measure droplets and add them to the droplet list'''
        newDroplets = []
        for d in droplets:
            pos = d[0]
            dims = d[1]
            angle = d[2]
            newDroplets.append({'frame':frame, 'time':time,'dropNum':-1,'x':pos[0],'y':pos[1],'dpos':0,'v':0,'w':dims[0],'l':dims[1],'angle':angle})
        
        df = pd.DataFrame(newDroplets)
        df = self.assignDroplets(df)
        return df
           
    def getEllipses(self, stream:cv.VideoCapture) -> int:
        '''get info about the ellipses. Returns 1 when video is done. Returns 0 to continue grabbing.'''
        grabbed, frame = stream.read() # read first frame
        if not grabbed:
            return 1 # video is done
        
        droplets = self.getDroplets(frame)
        time, frame = streamInfo(stream)
        self.prevDroplets = self.measureDroplets(droplets, time, frame) # save this timestep 
        self.dropletTab = pd.concat([self.dropletTab, self.prevDroplets], ignore_index=True) # add to the complete list

        return 0 
    
    def readFrames(self, startFrame:int=0, endFrame:int=100000, reportFreq:int=100) -> None:
        '''iterate through frames'''
        file = self.file
        stream = cv.VideoCapture(file)
        self.prevDroplets = self.emptyDropTab.copy()
        self.dropletTab = self.dropletTab[(self.dropletTab['frame']<=startFrame)|(self.dropletTab['frame']>endFrame)]
        stream.set(1, startFrame-1) # set to initial frame
        endFrame = min(endFrame, self.totalFrames)
        ret = 0
        frames = startFrame
        while ret==0 and frames<=endFrame:
            if frames%reportFreq==0 or frames==startFrame:
                logging.info(f'Analyzing frame {frames}/{endFrame}, [{self.totalFrames}]')
            try:
                ret = self.getEllipses(stream)
                frames+=1
            except Exception as e:
                logging.error(f'Error on frame {frames}: {e}')
                traceback.print_exc(limit=None, file=None, chain=True)
                return
        stream.release() # close the file
        self.finalFrame = max(self.finalFrame, frames)
        return 
    
    

    #---------------------------------------------------------------------
    


    def consolidateDroplets(self) -> pd.DataFrame:
        '''go through the list of droplets generated by readFrames and connect droplets that go offscreen'''

        if len(self.dropletTab)==0:
            return

        dropNums = self.dropletTab['dropNum'].unique()
        if -1 in dropNums:
            frames = list(self.dropletTab[self.dropletTab['dropNum']==-1]['frame'])
            logging.warning(f'Unlabeled droplets present in frames: {frames}')

        relabeledDroplets = self.dropletTab.copy()

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
            x0 = df.iloc[0]['x']
            y0 = df.iloc[0]['y']
            xf = df.iloc[-1]['x']
            yf = df.iloc[-1]['y']
            frame0 = df.iloc[0]['frame']
            framef = df.iloc[-1]['frame']
            endpoints.append({'dropNum':num, 'firstY':firstY, 'lastY':lastY, 'vinit':vinit, 'vfinal':vfinal, 'tinit':tinit, 'tfinal':tfinal, 'area':area, 'x0':x0, 'y0':y0, 'xf':xf, 'yf':yf, 'frame0':frame0, 'framef':framef})
        endpoints = pd.DataFrame(endpoints)
        self.endpoints = endpoints
        for i, row in endpoints.iterrows():
            if i>0:
                index = endpoints.loc[i].name
                iprev = i-1
                while iprev>=0:
                    try:
                        if endMatch(endpoints.loc[i], endpoints.loc[iprev]):
                            # opposite sign velocities and start/end on same edge and enter after the droplet left and area within 25%: same droplet
                            oldnum = endpoints.loc[i,'dropNum']
                            newnum = endpoints.loc[iprev,'dropNum']
                            relabeledDroplets.loc[relabeledDroplets['dropNum']==oldnum, 'dropNum']=newnum
                            endpoints.loc[i, 'dropNum']=newnum
                            endpoints.loc[i, 'vinit']=endpoints.loc[iprev]['vinit']
                            endpoints.loc[i, 'firstY']=endpoints.loc[iprev]['firstY']
                            endpoints.drop([iprev], inplace=True)
                            iprev=-1
                        else:
                            iprev=iprev-1
                    except:
                        iprev = iprev-1
                        
        self.relabeledDroplets = relabeledDroplets

        return relabeledDroplets

#---------------

    def baselineDroplets(self) ->Tuple[pd.DataFrame, dict]:
        '''determine baseline undeformed droplet dimensions'''
        if len(self.relabeledDroplets)==0:
            logging.info('baselineDroplets aborted. No data in relabeledDroplets')
            return [],{}

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
        baselines = pd.DataFrame(baselines)
        baselinesUnits = {'dropNum':'', 'w':'px', 'l':'px', 'r0':'px', 'w2':'px', 'l2':'px', 'r02':'px'}
        self.baselines = baselines
        self.baselinesUnits = baselinesUnits
        return baselines, baselinesUnits

    #---------------

    def splitTimes(self, profile) -> Tuple[pd.DataFrame, dict]:
        '''split the relabeled droplets into moves, using a Profile object from fileHandling'''

        if len(self.relabeledDroplets)==0:
            logging.info('splitTimes aborted. No data in self.relabeledDroplets')
            return [], {}

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

        moveTimes = pd.DataFrame(moveTimes)
        moveTimesUnits = {'gap':profile.units['gap'], 'rate':profile.units['rate'], 'direction':'', 'tprog':profile.units['time'], 't0':'s', 'tf':'s', 'vave':'px/s', 'vmed':'px/s', 'vave/rate':'px/rad', 'vmed/rate':'px/rad'}
        self.moveTimes = moveTimes
        self.moveTimesUnits = moveTimesUnits
        return moveTimes, moveTimesUnits
    
    #---------------
    
    
            
    



#---------------
        
def grecoK(lam:float) -> float:
    return (42496+191824*lam+261545*lam**2+111245*lam**3)/(60480*(1+lam)**3 )

    
def dropData(dropNum:int, df:pd.DataFrame, row:pd.Series, droplet, matrix, r0:float, mppx:float) -> dict:
    '''determine x and y values for each of the types of fits, as well as other droplet measurements'''
    t0 = row['t0']
    tf = row['tf']
    df0 = df[(df.time>=t0)&(df.time<=tf)] # points within this time range
    wave = df0.w.mean()*mppx # convert this to meters
    rz = wave/2
    lave = df0.l.mean()*mppx

    if wave>0 and lave>0:

        gdot = float(row['rate']) # shear rate in rad/s
        gdothz = gdot/(2*np.pi) # shear rate in hz
        etad = droplet.visc(gdot) # viscosity of droplet (Pa*s)
        etam = matrix.visc(gdot) # viscosity of matrix (Pa*s)
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
        if rzr0r0<0:
            srrrk = np.sqrt(-rzr0r0/k)
        else:
            srrrk = -1
        
        return {'dropNum':dropNum, 'gdot':gdot, 'gdotHz':gdothz, 'etad':etad, 'etam':etam, 'lam':lam, 'k':k, 'w':wave, 'l':lave, 'r0':r0, 'grem':gdotRetam, 'D':D, 'gr':gdotR, 'Dte':Dtayloretam, 'g2':g2, 'rzr0r0':rzr0r0, 'gem':gdotetam, 'srrrk':srrrk}
    
    else:
        return {}
    
#---------------

def summarizeDroplet(dTab:dropletTracker, dropNum:int, mppx:float, droplet, matrix) -> pd.DataFrame:
    '''summarize measurements of the given droplet number'''
    df = dTab.relabeledDroplets[dTab.relabeledDroplets.dropNum==dropNum]
    if len(df)==0:
        return
    baseline = dTab.baselines[dTab.baselines.dropNum==dropNum]
    if len(baseline)==0:
        logging.info(f'No baseline radius for droplet {dropNum}')
        return
    baseline = baseline.iloc[0]
    r0 = baseline['r0']*mppx # convert this to meters
    measurements = []
    for i,row in dTab.moveTimes.iterrows():
        dd = dropData(dropNum, df, row, droplet, matrix, r0, mppx)
        if len(dd)>0:
            measurements.append(dd)

    return pd.DataFrame(measurements)

def summarizeDroplets(dTab:dropletTracker, mppx:float, droplet, matrix) -> Tuple[pd.DataFrame, dict]:
    '''take final measurements for all droplets'''
    summary = pd.DataFrame()

    for dropNum in dTab.relabeledDroplets.dropNum.unique():
        s = summarizeDroplet(dTab, dropNum, mppx, droplet, matrix)
        summary = pd.concat([summary, s], ignore_index=True) # add to the complete list
    summaryUnits = {'dropNum':'', 'gdot':'rad/s', 'gdotHz':'1/s', 'etad':'Pa.s', 'etam':'Pa.s', 'lam':'', 'k':'', 'w':'m', 'l':'m', 'r0':'m', 'grem':'N/m', 'D':'', 'gr':'m/s', 'Dte':'m^2/(N*s)', 'g2':'1/s^2', 'rzr0r0':'1/m^2', 'gem':'N/m^2', 'srrrk':'1/m'}
    return summary, summaryUnits
       
#---------------

def linearReg(x:list, y:list, intercept:Union[float, str]='') -> Dict:
    '''Get a linear regression. y=bx+c'''
    y = np.array(y)
    X = np.array(x).reshape((-1,1))
    if type(intercept) is str:
        regr = LinearRegression().fit(X,y)
    else:
        y = y-intercept
        regr = LinearRegression(fit_intercept=False).fit(X,y)
    rsq = regr.score(X,y)
    c = regr.intercept_
    if not type(intercept) is str:
        c = c+intercept
    b = regr.coef_
    b = b[0]
    return {'b':b, 'c':c, 'rsq':rsq}
    
def sigmaFit(summ:pd.DataFrame, mode:int, intercept:Union[str,float]=0) -> dict:
    '''fit the summary data given a dataframe'''
    if mode==1:
        # method 1: Taylor discrete viscosity
        fit = linearReg(summ.gr, summ.Dte, intercept=intercept)
        sigma = 1/fit['b'] # N/m
    elif mode==2:
        # method 2: Taylor apparent viscosity
        lam = summ.lam.mean() # average lambda
        fit = linearReg(summ.grem, summ.D, intercept=intercept)
        sigma = 1/fit['b']*(19*lam+16)/(16*lam+16) # N/m
    elif mode==3:
        # method 3: Greco discrete viscosity
        summ = summ[summ.srrrk>0]
        fit = linearReg(summ.gem, summ.srrrk, intercept=intercept)
        sigma = 1/fit['b'] # N/m
    elif mode==4:
        # method 4: Greco apparent viscosity
        lam = summ.lam.mean() # average lambda
        etac = summ.etam.mean() # average matrix viscosity
        k = grecoK(lam)
        fit = linearReg(summ.g2, summ.rzr0r0, intercept=intercept)
        if fit['b']>0:
            sigma=''
        else:
            sigma = etac/np.sqrt(-fit['b']/k) # N/m  

    return {'mode':mode, 'sigma':sigma, 'slope':fit['b'], 'intercept':fit['c'], 'r2':fit['rsq']}


def sigmaFitX(summ:pd.DataFrame, mode:int, xmax:float=-1, xmin:float=-1, intercept:Union[str, float]=0) -> dict:
    '''fit the summary data using one of the 4 methods'''
    x = {1:'gr',2:'grem',3:'gem',4:'g2'}[mode]
    if xmax>0:
        summ = summ[(summ[x]<=xmax)]
    if xmin>0:
        summ = summ[(summ[x]>=xmin)]

    return sigmaFit(summ, mode, intercept=intercept)

#-------------------------------

def relaxSigma(d3:pd.DataFrame, mppx:float, p:float, etam:float, dropNum:int) -> dict:
    '''get surface tension for one time series d3'''
    if len(d3)<5:
        return {}
    dd = [(row['l']-row['w'])/(row['l']+row['w']) for i, row in d3.iterrows()] # Taylor D
    lnd = [np.log(i/dd[0]) for i in dd] # log of D/D0
    try:
        fit = linearReg(d3['time'], lnd)
    except:
        return {}
    b = fit['b']
    if b>0:
        return {}
    c = fit['c']
    r2 = fit['rsq']
    if r2<0.8:
        return {}
    w2 = d3.iloc[-1]['w']*mppx
    l2 = d3.iloc[-1]['l']*mppx
    r0 = ((w2/2)**2*(l2/2))**(1/3) # find r0
    sigma = -b*((2*p+3)*(19*p+16)*etam*r0)/(40*(p+1))
    return {'dropNum':dropNum, 'b':b, 'c':c, 'r2':r2, 'w2':w2, 'l2':l2, 'r0':r0, 'sigma':sigma}

def getRelaxation(dTab:dropletTracker, mppx:float, droplet, matrix) -> Tuple[pd.DataFrame, dict]:
    '''determine the surface tension via relaxation. Son, Y., & Migler, K. B. (2002). Interfacial tension measurement between immiscible polymers: Improved deformed drop retraction method. Polymer, 43(10), 3001–3006. https://doi.org/10.1016/S0032-3861(02)00097-6'''
    vcrit = 2
    d2 = dTab.relabeledDroplets[(dTab.relabeledDroplets.v<vcrit)&(dTab.relabeledDroplets.v>-vcrit)] # only select velocities near zero
    d2 = d2[d2.time<dTab.moveTimes.loc[0,'t0']] # only select times before first move
    etad = droplet.visc(10**-10)
    etam = matrix.visc(10**-10)
    p = etad/etam
    retval = []
    retvalUnits = {'dropNum':'', 'b':'1/s', 'c':'', 'r2':'', 'w2':'m', 'l2':'m', 'r0':'m', 'sigma':'N/m'}
    for dropNum in d2.dropNum.unique():
        d3 = d2[d2.dropNum==dropNum]
        r = relaxSigma(d3, mppx, p, etam, dropNum)
        if len(r)>0:
            retval.append(r)
    return pd.DataFrame(retval, columns=retvalUnits.keys()), retvalUnits
    
    