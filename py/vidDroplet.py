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
from sklearn import metrics
from sklearn.cluster import KMeans

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
    
# pd.options.display.float_format = '{:,.3f}'.format
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

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

def flatten(t):
    return [item for sublist in t for item in sublist]

def dropletChange(oldrow:pd.Series, newrow:pd.Series) -> dict:
    '''determine change in droplet'''
    x = newrow['x'] # find position of new droplet
    y = newrow['y']
    xprev = oldrow['x']
    yprev = oldrow['y']
    dx = x-xprev
    dy = y-yprev
    dd = np.sqrt(dx**2+dy**2)*np.sign(dy)  # positive dd means droplet is traveling in positive y
    dframe = (newrow['frame']-oldrow['frame'])
    if dframe>0:
        ddpf = dd/dframe                       # distance moved per frame
        v = dd/(newrow['time']-oldrow['time']) # velocity
    else:
        ddpf = 0
        v = 0
    oldvest = oldrow['vest']
    newvest = newrow['vest']
    dvol = abs(newvest/oldvest - 1)        # change in estimated volume
    newindex = newrow.name
    oldindex = oldrow.name
    match = True
    dropNum = oldrow['dropNum']
    return {'dropNum':dropNum, 'newindex':newindex, 'oldindex':oldindex, 'dd':dd, 'dpos':ddpf, 'dx':dx, 'dframe':dframe, 'v':v, 'dvol':dvol}

def closest_node(node:Tuple, nodes:Tuple) -> int:
    '''Find closest point to list of points. https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points'''
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def endMatch(drop:pd.Series, dropprev:pd.Series) -> bool:
    '''do these endpoints match?'''
    dx = abs(dropprev['xf']-drop['x0'])
    if dx>200:
        # x too far apart. 
        return False
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


def labelTimeClustersK(d2:pd.DataFrame, k:int) -> Tuple[list, float]:
        '''group times into clusters given a number of clusters'''
        k_means = KMeans(n_clusters=k)
        X = np.array(d2.time).reshape(-1,1)
        model = k_means.fit(X)
        y_hat = k_means.predict(X)
        
        labels = k_means.labels_
#         ch = metrics.calinski_harabasz_score(X, labels)
        ch = metrics.silhouette_score(X, labels, metric = 'euclidean')
        return labels, ch
        
    
def labelTimeClusters(d2:pd.DataFrame, numstops:int) -> pd.DataFrame:
    '''given a number of stops, cluster the points by time'''
    labels = [[0 for i in range(len(d2))]]
    ch = [0]
    for i in range(2, numstops+2):
        l,c = labelTimeClustersK(d2, i)
        labels.append(l)
        ch.append(c)
    bestfit = ch.index(max(ch))
    d3 = d2.copy()
    d3['labels'] = labels[bestfit]
    return d3

def labelTimeContinuous(d2:pd.DataFrame) -> pd.DataFrame:
    '''find continuous runs in the data'''
    d3 = d2.copy()
    d3['labels'] = (d3.frame.diff(1) > 10).astype('int').cumsum()
    return d3

        
def clusterTimes(d2:pd.DataFrame, numstops:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''cluster movements by time, turn into dataframe'''
    if len(d2)<20:
        return d2, []
    
    # only use the most prominent droplet
    counts = d2.dropNum.value_counts()
    biggest = counts.idxmax()
    d2 = d2[d2.dropNum==biggest]
    
#     d2 = labelTimeClusters(d2, numstops) # label clusters of times
    d2 = labelTimeContinuous(d2)
    
    # get times of stopped droplets
    times = []
    for i in d2.labels.unique():
        d3 = d2[d2.labels==i]
        tmin = d3.time.min()
        tmax = d3.time.max()
        tlen = tmax-tmin
        times.append({'label':i, 't0':tmin, 'tf':tmax, 'tlen':tlen})
    times = pd.DataFrame(times)
    return d2, times

#---------------


class dropletTracker:
    '''the purpose of this object is to store tables, where each row represents a droplet in a single frame'''
    
    def __init__(self, test, vidnum:int): 
        '''test is a Test object, defined in fileHandling'''
        self.file = test.videos[vidnum]
        self.bounds = {}
        self.emptyDropTab = pd.DataFrame(columns=['frame','time', 'dropNum', 'x', 'y', 'dpos', 'v', 'w', 'l', 'angle'])
        self.resetTables()
        self.dropletTabUnits = {'frame':'','time':'s', 'dropNum':'', 'x':'px', 'y':'px', 'dpos':'px/frame', 'v':'px/s', 'w':'px', 'l':'px', 'angle':'degree', 'vest':'px^3'}
        stream = cv.VideoCapture(self.file)
        self.fps = stream.get(cv.CAP_PROP_FPS)
        self.totalFrames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))
        stream.release()
        self.finalFrame = 0
        
    def resetTables(self) -> None:
        self.dropletTab = self.emptyDropTab.copy()
        self.prevDroplets = self.emptyDropTab.copy()

#---------------

    #---------------

    def findCritDD(self, newDroplets:pd.DataFrame) -> float:
        '''find the critical distance above which the new droplet must move'''
        critdd = 0
        dpos = self.prevDroplets[(self.prevDroplets.dpos>0)|(self.prevDroplets.dpos<0)] # only use nonzero velocities
        frame = newDroplets.iloc[0]['frame']
        dpos = dpos[dpos.frame>frame-5] # only use recent frames
        dave = dpos['dpos'].mean()      # average move distance per frame of previous droplets
        if abs(dave)<5 or len(dpos)==0:
            # change is too small. don't filter
            critdd = 0
        else:
            critdd = dave 
            # if the previous droplets were known to be moving,
            # the new move must be within bounds
        return critdd

    #---------------
                
    def changesTable(self, newDroplets:pd.DataFrame, critdd:float) -> pd.DataFrame:
        '''get a table of the changes between newdroplets and previous droplets'''
        changes = [[dropletChange(oldrow, newrow) for i,oldrow in self.prevDroplets.iterrows()] for j,newrow in newDroplets.iterrows()]
        changes = pd.DataFrame(flatten(changes))
        changes['dposnorm'] = changes['dpos']/critdd

        changes = changes[abs(changes.v)<10000]   # velocity small enough
        changes = changes[changes.dvol<0.25]     # change in volume small enough
        changes = changes[abs(changes.dx)<50]

        # filter 
        if abs(critdd)>0:
            changes = changes[(changes.dposnorm>0.25)|(abs(changes.dpos)<3)]       # movement large enough and in same direction or static
            changes = changes[changes.dposnorm<4]          # movement small enough
        
        return changes

    def updateDroplet(self, newDroplets:pd.DataFrame, row:pd.Series) -> None:
        '''update the drop num, given a row from the changes table'''
        oldrow = self.prevDroplets[self.prevDroplets.dropNum==row['dropNum']]
        oldrow = oldrow.iloc[0]
        for s in ['dropNum', 'dpos', 'v']:
            val = row[s]
            if np.sign(val)==np.sign(oldrow[s]) or oldrow[s]==0:
                newDroplets.loc[row['newindex'], s] = row[s] # assign droplet num to newDroplet row
            else:
                # sign is reversed, which should only happen for very small changes. set to 0 so it doesn't throw off critdd
                newDroplets.loc[row['newindex'], s] = 0
                
    def relabelFromMatch(self, changes:pd.DataFrame, newindex:int, dropNum:int) -> None:
        '''relabel dropletTab based on redundant matches during assignFromPrevious'''
        
        otherCand = changes[changes.newindex==newindex] # other matches that have the same index
        dndkeep = otherCand[otherCand.dropNum==dropNum]
        dpos = float(dndkeep.iloc[0]['dpos'])
        otherCand = otherCand[otherCand.dropNum!=dropNum]
        otherCand = otherCand[(otherCand.dvol<0.05)]    # very close volume
        otherCand = otherCand[(abs(otherCand.dpos)<3)|(abs(otherCand.dpos - dpos)/abs(dpos) < 0.1)]
            # haven't moved or within 25% of kept movement

        for i,row in otherCand.iterrows():
            oldDropNum = row['dropNum'] # drop number to be overwritten
            self.dropletTab["dropNum"].replace({oldDropNum: dropNum}, inplace=True)   # replace droplet numbers in droplet tabl
            self.prevDroplets = self.prevDroplets[self.prevDroplets.dropNum!=oldDropNum] # remove this droplet from prevDroplets            

    def assignFromPrevious(self, newDroplets:pd.DataFrame, diag:int=0) -> pd.DataFrame:
        '''assign new droplet numbers from previous droplet numbers'''
        if len(newDroplets)==0 or len(self.prevDroplets)==0:
            return newDroplets
        
        critdd = self.findCritDD(newDroplets)
        changes = self.changesTable(newDroplets, critdd)
        
        if len(changes)==0:
            return newDroplets
        
        if critdd>0:
            changes['fit'] = abs(changes['dposnorm']-1)/max(abs(changes['dposnorm']-1)) + changes['dvol']/max(changes['dvol'])
        else:
            changes['fit'] = changes['dvol']/max(changes['dvol'])

        if diag>0:
            print(newDroplets.iloc[0]['frame'], critdd)
            display(changes)
            
        if len(changes.newindex.unique())==len(changes):
            # 1 entry per new index
            for i,row in changes.iterrows():
                self.updateDroplet(newDroplets, row)
        else:
            # multiple possibilities per new index
            changes.sort_values(by='fit', inplace=True) # sort by volume difference and difference between expected and real movement
            i = 0
#             if diag>0:
#                 print(newDroplets.iloc[0]['frame'], critdd)
#                 display(changes)
            while len(changes)>0 and i<=max(changes.index):
                if i in changes.index:
                    row = changes.loc[i]
                    self.updateDroplet(newDroplets, row)
                    self.relabelFromMatch(changes, row['newindex'], row['dropNum'])
                    changes = changes[(changes.newindex!=row['newindex'])] # remove this index from the running
                i = i+1

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

    def assignDroplets(self, newDroplets:pd.DataFrame, diag:int=0) -> pd.DataFrame:
        '''assign droplet numbers'''
        # match droplets to previous droplets
        newDroplets = self.assignFromPrevious(newDroplets, diag=diag)
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
        interfaces2 = eliminateTouching(interfaces, bounds, dr=-10) # get rid of any blobs touching the aperture
        droplets = detectEllipses(interfaces2, diag=False)              # measure ellipses
        if diag:
            ellipse1 = drawEllipses(removed, droplets, diag=False)
            imshow(frame,ellipse1, interfaces2)
        return droplets
    
    def measureDroplets(self, droplets:List[tuple], time:float, frame:int, diag:int=0) -> None:
        '''measure droplets and add them to the droplet list'''
        newDroplets = []
        for d in droplets:
            pos = d[0]
            dims = d[1]
            angle = d[2]
            if dims[0]<(self.bounds['xf']-self.bounds['x0'])/2 and dims[1]<(self.bounds['yf']-self.bounds['y0'])/2 and pos[0]>0 and pos[1]>0:
                # filter out very large or out of frame droplets
                vest = 4/3*np.pi*dims[0]**2*dims[1]
                newDroplets.append({'frame':frame, 'time':time,'dropNum':-1,'x':pos[0],\
                                    'y':pos[1],'dpos':0,'v':0,\
                                    'w':dims[0],'l':dims[1],'angle':angle, 'vest':vest})
        
        df = pd.DataFrame(newDroplets)
        if len(df)>0:
            df = self.assignDroplets(df, diag=diag)
        return df
    
    def concatPrev(self, newDroplets) -> None:
        '''add new droplets to list of prevdroplets'''
        dn = newDroplets.dropNum.unique()
        savedDroplets = self.prevDroplets[[row['dropNum'] not in dn for i,row in self.prevDroplets.iterrows()]]
        self.prevDroplets = pd.concat([savedDroplets, newDroplets])
        self.prevDroplets.sort_values(by='y', inplace=True, ascending=False)
        self.prevDroplets.reset_index(drop=True, inplace=True)
           
    def getEllipses(self, stream:cv.VideoCapture, diag:int=0) -> int:
        '''get info about the ellipses. Returns 1 when video is done. Returns 0 to continue grabbing.'''
        grabbed, frame = stream.read() # read first frame
        if not grabbed:
            return 1 # video is done
        
        droplets = self.getDroplets(frame)
        time, frame = streamInfo(stream)
        newDroplets = self.measureDroplets(droplets, time, frame, diag=diag-1) # save this timestep 
        if len(newDroplets)>0:
            
            
            if len(self.prevDroplets)>0:
                vmean = newDroplets.v.mean()
                pd0 = self.prevDroplets[(self.prevDroplets.v>0)|(self.prevDroplets.v<0)]
                if len(pd0)>0:
                    vprev = pd0.v.mean()
                else:
                    # if all droplets from previous step were new, still keep the old droplets
                    vprev = vmean
                if abs(vmean)<10 or abs(vmean - vprev)<(abs(vprev+vmean)):
                    # static or moving in same direction. keep old droplets that weren't detected
                    self.concatPrev(newDroplets)
                else:
                    if diag>0:
                        print(frame, 'reset 1', vmean, vprev, abs(vmean - vprev))
                    # change in movement. turn off critdd for the next step
                    self.prevDroplets. v = [0 for i in range(len(self.prevDroplets))]
                    self.prevDroplets.dpos = [0 for i in range(len(self.prevDroplets))]
            else:
                if diag>0:
                    print(frame, 'reset 2')
                # don't keep old droplets
                self.prevDroplets = newDroplets
            self.dropletTab = pd.concat([self.dropletTab, newDroplets], ignore_index=True) # add to the complete list
        else:
            fprev = self.dropletTab[self.dropletTab.frame<frame]
            if frame - fprev.frame.max() > 50:
                # more than _ skipped frames. get rid of old
                self.prevDroplets = newDroplets

        return 0 
    
    def initializePrev(self, startFrame:int) -> None:
        '''initialize previous droplets based on existing table'''
        dt0 = self.dropletTab.copy()
        dt0 = dt0[dt0.frame<startFrame]  # only use table before start frame
        self.prevDroplets = self.emptyDropTab.copy()
        for dn in dt0.dropNum.unique():
            dt1 = dt0[dt0.dropNum==dn]   # for each droplet number, take the last frame
            maxframe = dt1.frame.max()
            if maxframe>startFrame-50:   # only use recent droplets
                dt1 = dt1[dt1.frame==maxframe]
                self.prevDroplets = pd.concat([self.prevDroplets,dt1])
        self.prevDroplets.reset_index(inplace=True, drop=True)
    
    def readFrames(self, startFrame:int=0, endFrame:int=100000, reportFreq:int=100, diag:int=1) -> None:
        '''iterate through frames'''
        file = self.file
        stream = cv.VideoCapture(file)
        self.dropletTab = self.dropletTab[(self.dropletTab['frame']<startFrame)|(self.dropletTab['frame']>endFrame)]
        self.initializePrev(startFrame) # initialize table of previous droplets
        stream.set(1, startFrame-1) # set to initial frame
        endFrame = min(endFrame, self.totalFrames)
        ret = 0
        frames = startFrame
        while ret==0 and frames<=endFrame:
            if (frames%reportFreq==0 or frames==startFrame) and diag>0:
                logging.info(f'Analyzing frame {frames}/{endFrame}, [{self.totalFrames}]')
            try:
                ret = self.getEllipses(stream, diag=diag-1)
                frames+=1
            except Exception as e:
                logging.error(f'Error on frame {frames}: {e}')
                traceback.print_exc(limit=None, file=None, chain=True)
                return
        stream.release() # close the file
        self.finalFrame = max(self.finalFrame, frames)
        self.dropletTab.sort_values(by='frame', inplace=True)
        if diag>1:
            display(self.dropletTab[(self.dropletTab['frame']>=startFrame)&(self.dropletTab['frame']<=endFrame)])
        return 
    
    

    #---------------------------------------------------------------------
    


    def consolidateDroplets(self) -> pd.DataFrame:
        '''go through the list of droplets generated by readFrames and connect droplets that go offscreen'''

        if len(self.dropletTab)==0:
            self.relabeledDroplets = self.dropletTab
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
    
    def splitTimesSingle(self, profile, times) -> pd.DataFrame:
        '''if there is only one relaxation step'''

        moveTimes = pd.DataFrame(columns=['gap', 'rate', 'direction', 'tprog', 't0', 'tf', 'vave', 'vmed', 'vave/rate', 'vmed/rate'], index = profile.table.index)
        for s in ['gap', 'rate', 'direction']:
            moveTimes[s] = profile.table[s]
        moveTimes['tprog'] = profile.table['time']
        
        longest = times.tlen.idxmax()
        for s in ['t0', 'tf']:
            moveTimes.loc[2, s] = times.loc[longest, s]
        moveTimes.loc[0, 't0']=0
        moveTimes.loc[0, 'tf']=times.loc[longest, 't0']/2
        moveTimes.loc[1,'t0']=times.loc[longest, 't0']/2
        moveTimes.loc[1,'tf']=times.loc[longest, 't0']
        return moveTimes
        
    
    def splitTimesRelax(self, profile) -> pd.DataFrame:
        '''split the relabeled droplets into moves, where there are relaxation steps'''
        if len(self.relabeledDroplets)==0:
            logging.info('splitTimes aborted. No data in self.relabeledDroplets')
            return [], {}

        df = self.relabeledDroplets
        vcrit = 2
        d2 = df[(df.v<vcrit)&(df.v>-vcrit)] # static droplets
        if len(d2)<20:
            logging.warning('splitTimesRelax failed to find static droplets')
            return []
#         numstops = len(profile.table['mode']=='relax')
        numstops = 20
        d2, times = clusterTimes(d2, numstops)
        
        if len(profile.table==3):
            return self.splitTimesSingle(profile, times)
        
        moveTimes = pd.DataFrame(columns=['gap', 'rate', 'direction', 'tprog', 't0', 'tf', 'vave', 'vmed', 'vave/rate', 'vmed/rate'], index = profile.table.index)
        for s in ['gap', 'rate', 'direction']:
            moveTimes[s] = profile.table[s]
        moveTimes['tprog'] = profile.table['time']
        movei = 1
        if times.loc[0, 't0']<0.1:
            # first stop is before start of run
            moveTimes.loc[0, 't0']=times.loc[0, 'tf']
        else:
            moveTimes.loc[0, 't0']=0
            
        progtimes = [float(i) for i in moveTimes.tprog]
            
        # go through rows in the stopped actual times, and find the corresponding row in moveTimes
        scale = 0
        for i,trow in times.iterrows():
            if i>0:
                t0 = trow['t0']
                tf = trow['tf']
                d3 = d2[d2.labels==trow['label']]
                vave = d3.v.mean()
                vmed = d3.v.median()
                stop = False
                while movei<len(progtimes) and not stop:
                    progdt = sum(progtimes[0:movei])
                    dt = (t0-moveTimes.loc[0, 't0'])
                    if moveTimes.loc[movei, 'rate']==0 and (progtimes[movei]>trow['tlen'] or movei==len(moveTimes)-1) and progdt>dt:
                        # programmed relax row and
                        # programmed time is longer than move time and 
                        # elapsed time since start is shorter than programmed time
                        # (actual move times are shorter than programmed times)
                        stop = True
                        moveTimes.loc[movei, ['t0', 'tf', 'vave', 'vmed']] = [t0, tf, vave, vmed]
                        if scale==0:
                            scale1 = progdt/dt # programmed change in time vs actual change in time
                            scale2 = progtimes[movei]/trow['tlen'] # programmed length vs actual length
                            scale = (scale1+scale2)/2
                    movei = movei+1
                if movei==len(progtimes) and not stop:
                    moveTimes = moveTimes.append({'gap':float(moveTimes.loc[0, 'gap']), 'rate':0, 'tprog':tf-t0, 't0':t0, 'tf':tf, 'vave':vave, 'vmed':vmed}, ignore_index=True)
        
        
        
        if scale==0:
            # if nothing was collected, revert to the conventional method
            logging.warning('splitTimesRelax failed to correlate stopped droplets to profile')
            return []
        
        # fill in remaining times
        for i,mrow in moveTimes.iterrows():
            if i>0 and pd.isnull(mrow['t0']):
                mrow['t0'] = moveTimes.loc[0,'t0'] + sum(progtimes[0:i])/scale
        for i,mrow in moveTimes.iterrows():
            if pd.isnull(mrow['tf']):
                if i<len(moveTimes)-1:
                    mrow['tf'] = moveTimes.loc[i+1,'t0']
                else:
                    mrow['tf'] = df.time.max()
            d4 = df[(df.time>mrow['t0'])&(df.time<mrow['tf'])]
            if len(d4)>0:
                mrow['vave'] = d4.v.mean()
                mrow['vmed'] = d4.v.median()
                fr = float(mrow['rate'])
                if fr>0:
                    mrow['vave/rate'] = abs(mrow['vave']/fr)
                    mrow['vmed/rate'] = abs(mrow['vmed']/fr)
        
        return moveTimes
    
    def splitTimesOsc(self, profile) -> pd.DataFrame:
        '''split the relabeled droplets into moves, where the profile is just back and forth, using a Profile object from fileHandling'''
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
            vcrit = 10
            if row['direction']=='cw':
                # velocity should be positive
                dfrange = dfrange[(dfrange.v>vcrit)]
            elif row['direction']=='acw':
                dfrange = dfrange[(dfrange.v<-vcrit)]
            else:
                dfrange = dfrange[(dfrange.v>-vcrit)&(dfrange.v<vcrit)]
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
            if row['rate']==0:
                vavescale = 0
                vmdescale = 0
            else:
                try:
                    vavescale = abs(vave/float(row['rate']))
                    vmedscale = abs(vmed/float(row['rate']))
                except:
                    vavescale = 0
                    vmedscale = 0
            dropStat = {'t0':firstTime, 'tf':finalTime, 'vave':vave, 'vmed':vmed, 'vave/rate':vavescale,'vmed/rate':vmedscale} # info from relabeledDroplets
            keep = {**rowkeep, **dropStat} 
            moveTimes.append(keep)

        moveTimes = pd.DataFrame(moveTimes)
        
        
        return moveTimes
        

    def splitTimes(self, profile) -> Tuple[pd.DataFrame, dict]:
        '''split the relabeled droplets into moves, using a Profile object from fileHandling'''

        if len(self.relabeledDroplets)==0:
            logging.info('splitTimes aborted. No data in self.relabeledDroplets')
            return [], {}
        
        if 'relax' in list(profile.table['mode']):
            moveTimes =  self.splitTimesRelax(profile)
        else:
            moveTimes =  self.splitTimesOsc(profile)
        moveTimesUnits = {'gap':profile.units['gap'], 'rate':profile.units['rate'], 'direction':'', 'tprog':profile.units['time'], 't0':'s', 'tf':'s', 'vave':'px/s', 'vmed':'px/s', 'vave/rate':'px/rad', 'vmed/rate':'px/rad'}
        self.moveTimes = moveTimes
        self.moveTimesUnits = moveTimesUnits
        return moveTimes, moveTimesUnits
    
    #---------------
    
    
            
    




#---------------

def removeOutliers(dfi:pd.DataFrame, col:str, low:float=0.05, high:float=0.95) -> pd.DataFrame:
    '''https://nextjournal.com/schmudde/how-to-remove-outliers-in-data'''
    df = dfi.copy()
    y = df[col]
    removed_outliers = y.between(y.quantile(low), y.quantile(high))
    index_names = df[~removed_outliers].index
    df.drop(index_names, inplace=True)
    return df

def grecoK(lam:float) -> float:
    return (42496+191824*lam+261545*lam**2+111245*lam**3)/(60480*(1+lam)**3 )


    
def dropData(dropNum:int, df:pd.DataFrame, row:pd.Series, droplet, matrix, r0:float, mppx:float, diag:int=0) -> dict:
    '''determine x and y values for each of the types of fits, as well as other droplet measurements'''
    t0 = row['t0']
    tf = row['tf']
    df0 = df[(df.time>=t0)&(df.time<=tf)] # points within this time range
    df0 = df0[abs(df0.v)>0]
    df0 = removeOutliers(df0, 'v', high=0.9, low=0.1) # remove jagged moves
    if df0.v.mean()<0:
        df0 = df0[df0.v<0] # if the average movement is negative, only include points where movement is negative
    else:
        df0 = df0[df0.v>0] # if the average movement is positive, only include points where movement is positive
    if df0.x.max() - df0.x.min() > 100:
        # big x range. might contain disconnected parts
        dx = [0]+[df0.iloc[i]['x']-df0.iloc[i-1]['x'] for i in range(1, len(df0))]  # change in x list
        xshifts = [0]+[i for i in range(len(dx)) if abs(dx[i])>50]+[len(df0)]       # locations of big changes
        dflist = [df0.iloc[xshifts[i]:xshifts[i+1]] for i in range(len(xshifts)-1)] # break df0 into sublists at changes
        lens = [len(l) for l in dflist]              # lengths of sublists
        lpos = lens.index(max(lens))                 # position of longest sublist
        df0 = dflist[lpos]                           # take longest sublist
    
    if len(df0)<5:
        return {}
    
    if diag>0:
        # plot trajectory of droplet
        plt.scatter(df0['x'], df0['y'], s=3)
        plt.plot(df0['x'], df0['y'], label=f'{dropNum}, {row.name}')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    if diag>1:
        # display all points for aggressive diagnostics
        display(df0)
    
    wave = df0.w.mean()*mppx # convert this to meters
    rz = wave/2
    lave = df0.l.mean()*mppx
    vave = df0.v.mean()*mppx # velocity in m/s
    gdothzest = abs(vave/(float(row['gap'])/10**6)) # estimated shear rate in Hz based on velocity of droplet
    
    if not (wave>0 and lave>0):
        return {}

    gdotrad = float(row['rate'])*(2*np.pi) # shear rate in rad/s
    gdothz = float(row['rate']) # shear rate in hz
    
    if gdothz==0:
        return {}

    etad = droplet.visc(gdothz) # viscosity of droplet (Pa*s)
    etam = matrix.visc(gdothz) # viscosity of matrix (Pa*s)
    lam = etad/etam # viscosity ratio
    k = grecoK(lam) # Greco eq, unitless

    # Taylor method apparent viscosity: slope = 1/sigma*(19lambda+16)/(16lambda+16)
    gdotRetam = gdothzest*r0*etam # Pa*m
    D = (lave-wave)/(lave+wave)

    # Taylor method discrete viscosities: slope = 1/sigma
    gdotR = gdothzest*r0 # (m/s)
    Dtayloretam = D*(16*lam+16)/((19*lam+16)*etam) # 1/(Pa*s)

    # Greco method apparent viscosity: slope = k(etam/sigma)^2
    g2 = gdothzest**2 # (1/s^2)
    rzr0r0 = (rz-r0)/r0**3

    # Greco method discrete viscosities: slope = 1/sigma
    gdotetam = gdothzest*etam # Pa*m/s
    if rzr0r0<0:
        srrrk = np.sqrt(-rzr0r0/k)
    else:
        srrrk = -1
        
    if gdothz>0:
        gdoterr = abs(abs(gdothzest/gdothz)-1) # fraction error from intended rate
    else:
        gdoterr = gdotHzest

    return {'dropNum':dropNum, 'N':len(df0), 'gdotrad':gdotrad, 'gdotHz':gdothz, 'etad':etad, 'etam':etam, 'lam':lam, 'k':k, 'w':wave, 'l':lave, 'v':vave, 'gdotHzest':gdothzest, 'gdotHzerr':gdoterr, 'r0':r0, 'grem':gdotRetam, 'D':D, 'gr':gdotR, 'Dte':Dtayloretam, 'g2':g2, 'rzr0r0':rzr0r0, 'gem':gdotetam, 'srrrk':srrrk}

        
    
#---------------

def summarizeDroplet(dTab:dropletTracker, dropNum:int, mppx:float, droplet, matrix, diag:int=0) -> pd.DataFrame:
    '''summarize measurements of the given droplet number'''
    if len(dTab.moveTimes)==0:
        return []
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
        # for each pass, summarize the droplet data into one row
        dd = dropData(dropNum, df, row, droplet, matrix, r0, mppx, diag=diag)
        if len(dd)>0:
            measurements.append(dd)
            
    if diag>0:
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        ax = plt.gca()
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.title('Droplet number, pass number')

    return pd.DataFrame(measurements)

def summarizeDroplets(dTab:dropletTracker, mppx:float, droplet, matrix, diag:int=0) -> Tuple[pd.DataFrame, dict]:
    '''take final measurements for all droplets'''
    summary = pd.DataFrame()

    for dropNum in dTab.relabeledDroplets.dropNum.unique():
        s = summarizeDroplet(dTab, dropNum, mppx, droplet, matrix, diag=diag)
        if len(s)>0:
            summary = pd.concat([summary, s], ignore_index=True) # add to the complete list
    gap = float(dTab.moveTimes.loc[0, 'gap'])*10**-6        
#     summary = summary[summary.gdotHzerr<0.5] # filter by speed accuracy
    summary = summary[summary.r0<gap/2] # filter by droplet size
    summaryUnits = {'dropNum':'', 'N':'', 'gdotrad':'rad/s', 'gdotHz':'1/s', 'etad':'Pa.s', 'etam':'Pa.s', 'lam':'', 'k':'', 'w':'m', 'l':'m', 'v':'m/s', 'gdotHzest':'1/s', 'gdotHzerr':'', 'r0':'m', 'grem':'N/m', 'D':'', 'gr':'m/s', 'Dte':'m^2/(N*s)', 'g2':'1/s^2', 'rzr0r0':'1/m^2', 'gem':'N/m^2', 'srrrk':'1/m'}
    return summary, summaryUnits
       
#---------------

def polyfit(x:List[float], y:List[float], degree:int) -> Dict:
    '''fit polynomial'''
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r2'] = ssreg / sstot
    results['coeffs'] = list(coeffs)

    return results

def quadReg(x:list, y:list) -> Dict:
    '''quadratic regression'''
    res = polyfit(x,y,2)
    return {'a':res['coeffs'][0], 'b':res['coeffs'][1], 'c':res['coeffs'][2], 'r2':res['r2']}


def linearReg(x:list, y:list, intercept:Union[float, str]='') -> Dict:
    '''Get a linear regression. y=bx+c'''
    if len(y)<5:
        return {}
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
        if len(fit)==0:
            return {}
        sigma = 1/fit['b'] # N/m
    elif mode==2:
        # method 2: Taylor apparent viscosity
        lam = summ.lam.mean() # average lambda
        fit = linearReg(summ.grem, summ.D, intercept=intercept)
        if len(fit)==0:
            return {}
        sigma = 1/fit['b']*(19*lam+16)/(16*lam+16) # N/m
    elif mode==3:
        # method 3: Greco discrete viscosity
        summ = summ[summ.srrrk>0]
        fit = linearReg(summ.gem, summ.srrrk, intercept=intercept)
        if len(fit)==0:
            return {}
        sigma = 1/fit['b'] # N/m
    elif mode==4:
        # method 4: Greco apparent viscosity
        lam = summ.lam.mean() # average lambda
        etac = summ.etam.mean() # average matrix viscosity
        k = grecoK(lam)
        fit = linearReg(summ.g2, summ.rzr0r0, intercept=intercept)
        if len(fit)==0:
            return {}
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

def plotRelaxFitDDR(fit:dict, x:list, y:list, xfull:list, yfull:list, vapp:list, r0:float, display:bool=True) -> None:
    '''plot the fit on data. x and y are for the selected data. xfull and yfull are for the whole time series. vapp is apparent volume for whole series. r0 is final radius'''
    fig,ax = plt.subplots(1,1)
    ax.scatter(xfull, yfull, color='#949494', s=0.3)
    ax.scatter(x,y, color='#c98647')
    xlist = [min(x), max(x)]
    ylist = [fit['b']*i+fit['c'] for i in xlist]
    
    ax.plot(xlist, ylist, color='black')
    ax.set_ylabel('ln(D)', color='#c98647')
    ax.set_xlabel('time (s)')
    ax2 = ax.twinx()
    ax2.scatter(xfull, vapp, color='#475fc9', s=0.3)
    ax2.set_ylabel('apparent dimensionless volume', color='#475fc9')
    x0 = 0.6
    ax.text(x0,0.9, "$\sigma$={:.2f} mJ/m$^2$".format(1000*fit['sigma']), transform=ax.transAxes)
    ax.text(x0,0.8, "$r^2$={:.2f}".format(fit['r2']), transform=ax.transAxes)
    ax.text(x0,0.7, "$r0$={:.0f} um".format(10**6*r0), transform=ax.transAxes)
    ax.text(x0,0.6, "Droplet {:.0f}".format(fit['dropNum']), transform=ax.transAxes)
    if not display:
        plt.close()
    return fig


def relaxSigmaDDR(d3:pd.DataFrame, mppx:float, p:float, etam:float, dropNum:int, diag:bool=False) -> dict:
    '''get surface tension for one time series d3'''
    if len(d3)<5:
        return {}, []
    d4 = d3.copy()
    w2 = d3.iloc[-1]['w']*mppx
    l2 = d3.iloc[-1]['l']*mppx
    r0 = ((w2/2)**2*(l2/2))**(1/3) # find r0
    v0 = 4/3*np.pi*r0**3
    d4['vapp'] = [(np.pi*row['l']*row['w']**2*mppx**3)/(6*v0) for i, row in d3.iterrows()] # apparent normalized volume
    d4['lcalc'] = [8*(r0/mppx)**3/(row['w'])**2 for i, row in d4.iterrows()] # calculate L from volume conservation
#     dd = [(row['l']-row['w'])/(row['l']+row['w']) for i, row in d3.iterrows()] # Taylor D
    d4['D'] = [(row['lcalc']-row['w'])/(row['lcalc']+row['w']) for i, row in d4.iterrows()] # Taylor D
    d4 = d4[d4.D>0]
    lnd = [np.log(i) for i in d4['D']] # log of D/D0
    times = list(d4['time'])
    try:
        num = len(times)
        fit = {'rsq':0}
        while num>=10 and fit['rsq']<0.97:
            times1 = times[:num]
            lnd1 = lnd[:num]
            fit = linearReg(times1, lnd1)
            num = int(num*0.9)
    except Exception as e:
        return {}, []
    if len(fit)==0 or not 'b' in fit:
        return {}, []
    b = fit['b']
    if b>0:
        return {}, []
    c = fit['c']
    r2 = fit['rsq']
    sigma = -b*((2*p+3)*(19*p+16)*etam*r0)/(40*(p+1))
    retval = {'dropNum':dropNum, 'b':b, 'c':c, 'r2':r2, 'w2':w2, 'l2':l2, 'r0':r0, 'p':p, 'sigma':sigma, 'method':'DDR'}
    fig = plotRelaxFitDDR(retval, times1, lnd1, times, lnd, list(d4['vapp']), r0, display=diag)
    return retval, fig

def IFRf(x:float) -> float:
    xx1 = 1+x+x**2
    if xx1<0:
        return -1
    b = np.sqrt(xx1)/(1-x)
    if b<0:
        return -1
    try:
        f = 3/2*np.log(b)+3**1.5/2*np.arctan(np.sqrt(3)*x/(2+x))-x/2-4/x**2
    except:
        f = -1
    return f

def plotRelaxFitIFR(fit:dict, x:list, y:list, xfull:list, yfull:list, r0:float, display:bool=True) -> None:
    '''plot the fit on data. x and y are for the selected data. xfull and yfull are for the whole time series. vapp is apparent volume for whole series. r0 is final radius'''
    fig,ax = plt.subplots(1,1)
    ax.scatter(xfull, yfull, color='#949494', s=0.3)
    ax.scatter(x,y, color='#c98647')
    xlist = [min(x), max(x)]
    ylist = [fit['b']*i+fit['c'] for i in xlist]
    
    ax.plot(xlist, ylist, color='black')
    ax.set_ylabel('f(R/R0)', color='#c98647')
    ax.set_xlabel('time (s)')
    x0 = 0.6
    ax.text(x0,0.9, "$\sigma$={:.2f} mJ/m$^2$".format(1000*fit['sigma']), transform=ax.transAxes)
    ax.text(x0,0.8, "$r^2$={:.2f}".format(fit['r2']), transform=ax.transAxes)
    ax.text(x0,0.7, "$r0$={:.0f} um".format(10**6*r0), transform=ax.transAxes)
    ax.text(x0,0.6, "Droplet {:.0f}".format(fit['dropNum']), transform=ax.transAxes)
    if not display:
        plt.close()
    return fig

def relaxSigmaIFR(d3:pd.DataFrame, mppx:float, p:float, etam:float, dropNum:int, diag:bool=False) -> dict:
    '''get surface tension for one time series d3'''
    if len(d3)<5:
        return {}, []

    w2 = d3.iloc[-1]['w']*mppx
    l2 = d3.iloc[-1]['l']*mppx
    r0 = ((w2/2)**2*(l2/2))**(1/3) # find r0
    d4 = d3.copy()
    d4['f'] = [IFRf(row['w']*mppx/(2*r0)) for i, row in d3.iterrows()]
    d4 = d4[d4.f>0]
    times = list(d4['time'])
    flist = list(d4['f'])
    
    try:
        num = len(times)
        fit = {'rsq':0}
        while num>=10 and fit['rsq']<0.97:
            times1 = times[:num]
            flist1 = flist[:num]
            fit = linearReg(times1, flist1)
            num = int(num*0.9)
    except Exception as e:
        print(e)
        return {}, []
    if len(fit)==0 or not 'b' in fit:
        return {}, []
    b = fit['b']
#     if b<0:
#         return {}, []
    c = fit['c']
    r2 = fit['rsq']
    sigma = b*((1.7*p+1)*etam*r0)/(2.7)
    retval = {'dropNum':dropNum, 'b':b, 'c':c, 'r2':r2, 'w2':w2, 'l2':l2, 'r0':r0, 'p':p, 'sigma':sigma, 'method':'IFR'}
    fig = plotRelaxFitIFR(retval, times1, flist1, times, flist, r0, display=diag)
    return retval, fig

def getRelaxation(dTab:dropletTracker, mppx:float, droplet, matrix, diag:bool=False) -> Tuple[pd.DataFrame, dict, list]:
    '''determine the surface tension via relaxation via DDR. Son, Y., & Migler, K. B. (2002). Interfacial tension measurement between immiscible polymers: Improved deformed drop retraction method. Polymer, 43(10), 3001–3006. https://doi.org/10.1016/S0032-3861(02)00097-6 '''
    
    vcrit = 50
    d1 = dTab.relabeledDroplets[(dTab.relabeledDroplets.v<vcrit)&(dTab.relabeledDroplets.v>-vcrit)] # only select velocities near zero
    
    if len(d1)<5:
        logging.warning('Not enough points for relaxation.')
        return [], {}, []
    
    etad = droplet.visc(0)
    etam = matrix.visc(0)
    p = etad/etam
    retval = []
    retvalUnits = {'dropNum':'', 'b':'1/s', 'c':'', 'r2':'', 'w2':'m', 'l2':'m', 'r0':'m', 'sigma':'N/m'}
    figlist = []
    
    if len(dTab.moveTimes)>0:
        if 0 in dTab.moveTimes.rate:
            times = dTab.moveTimes[dTab.moveTimes.rate==0]
        else:
            times = pd.DataFrame([{'t0':0, 'tf':dTab.moveTimes.loc[0,'t0']}])
    else:
        _, times = clusterTimes(d1, 20) # label clusters of times

    if len(times)==0:
        return [], {}, []
    
    for i,trow in times.iterrows():
        d2 = d1[(d1.time>trow['t0'])&(d1.time<trow['tf'])] # only select times before first move
        for dropNum in d2.dropNum.unique():
            d3 = d2[d2.dropNum==dropNum]
            r,fig = relaxSigmaDDR(d3, mppx, p, etam, dropNum, diag=diag)
            if len(r)>0:
                retval.append(r)
                figlist.append(fig)
            r2,fig2 = relaxSigmaIFR(d3, mppx, p, etam, dropNum, diag=diag)
            if len(r2)>0:
                retval.append(r2)
                figlist.append(fig2)  
    return pd.DataFrame(retval, columns=retvalUnits.keys()), retvalUnits, figlist






    
    