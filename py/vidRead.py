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

import re

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from vidDroplet import *
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

def plainIm(file:str, ic:Union[int, bool], checkUnits:bool=True) -> Tuple[Union[pd.DataFrame, List[Any]], Dict]:
    '''import a csv to a pandas dataframe. ic is the index column. Int if there is an index column, False if there is none. checkUnits=False to assume that there is no units row. Otherwise, look for a units row'''
    if os.path.exists(file):
        try:
            toprows = pd.read_csv(file, index_col=ic, nrows=2)
            toprows = toprows.fillna('')
            row1 = list(toprows.iloc[0])
            if checkUnits and all([type(s) is str for s in row1]):
                # row 2 is all str: this file has units
                unitdict = dict(toprows.iloc[0])
                skiprows=[1]
            else:
                unitdict = dict([[s,'undefined'] for s in toprows])
                skiprows = []
            d = pd.read_csv(file, index_col=ic, dtype=float, skiprows=skiprows)
        except Exception as e:
            print(e)
            return [],{}
        return d, unitdict
    else:
        return [], {}

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
        plt.close()
        
    def clean(self):
#         self.axs[1,1].legend(bbox_to_anchor=(1.05, 0), loc='lower left')
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
        ylist = [fit['slope']*i+fit['intercept'] for i in xlist]
        ax.plot(xlist, ylist, color='black')
        if fit['mode']==4:
            x0 = 0.6
        else:
            x0 = 0.02
        ax.text(x0,0.9, "$\sigma$={:.0f} mJ/m$^2$".format(1000*fit['sigma']), transform=ax.transAxes)
        ax.text(x0,0.8, "$r^2$={:.2f}".format(fit['r2']), transform=ax.transAxes)

#-----------------------------------


#-----------------------------------    

class vidInfo:
    '''holds tables of data about the video'''
    
    def __init__(self, test, vidnum:int):
        '''profile is a profile object, defined in fileHandling'''
        self.test = test
        self.file = test.videos[vidnum]
        self.profile = test.profile
        self.droplet = test.droplet
        self.matrix = test.matrix
        self.dTabs = dropletTracker(test, vidnum)
        self.relabeledDroplets = []
        self.mppx = 1/(cfg.vidRead.scale*1000)
        
        
    def detectDropletOneFrame(self, time:float=-1, frameNum:int=-1, diag:bool=False) -> List[tuple]:
        '''detect droplets for just one frame'''
        frame = readSpecificFrame(self.file, time=time, frameNum=frameNum)
        droplets = self.dTabs.getDroplets(frame, diag=diag)
        return droplets

    #---------------------------------


    def summarizeDroplets(self) -> None:
        self.summary, self.summaryUnits = summarizeDroplets(self.dTabs, self.mppx, self.droplet, self.matrix)
        
    def getSigma(self, plot:bool=False,  xminlist:dict={1:-1,2:-1,3:-1,4:-1}, xmaxlist:dict={1:-1,2:-1,3:-1,4:-1}, interceptlist:dict={1:0,2:0,3:0,4:0}) -> None:
        '''calculate sigma 4 ways'''
        fits = dict([[i+1,sigmaFitX(self.summary, i+1, xmax=xmaxlist[i+1], xmin=xminlist[i+1], intercept=interceptlist[i+1])] for i in range(4)])
        self.fits = fits
        self.sigmaUnits = {'mode':'', 'sigma':'N/m', 'slope':'m/N, m/N, m/N, s^2/m^2', 'intercept':'m^2/(N*s), , 1/m, 1/m^2', 'r2':''}
        if plot:
            for i in fits:
                self.summaryPlot.plotFit(fits[i])
        return fits   
    
    def getRelaxation(self) -> None:
        self.relaxation, self.relaxationUnits = getRelaxation(self.dTabs, self.mppx, self.droplet, self.matrix)
    
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
            df = self.dTabs.relabeledDroplets
        else:
            df = self.dTabs.dropletTab
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
        if len(df.dropNum)<8:
            axs[3].legend(bbox_to_anchor=(1.05, 0), loc='lower left')
        plt.close()
        return fig

    #--------------------------------------

    def plotSummaries(self, splitDroplets:bool=True, label:str='', getSigma:bool=False, export:bool=False) -> None:
        '''plot summary data'''
        self.summaryPlot = summaryPlot()
        df = self.summary
        if splitDroplets:
            for n in df.dropNum.unique():
                d1 = df[df['dropNum']==n]
                self.summaryPlot.plotSummary(d1, label=label+"_"+str(int(n)))
        else:
            self.summaryPlot.plotSummary(df, label=label)
        if getSigma:
            self.getSigma(plot=True) # calculate sigmas
        self.summaryPlot.clean()
    
    #----------------------------------
    
    def exportName(self) -> Tuple[str,str]:
        folder = cfg.path.export
        if not os.path.exists(folder):
            os.mkdir(folder)
        sample = self.droplet.name+'_'+self.matrix.name
        folder = os.path.join(folder, sample)
        if not os.path.exists(folder):
            os.mkdir(folder)
        times = re.split('_', os.path.splitext(os.path.basename(self.file))[0])
        name = sample+'_'+times[-2]+'_'+times[-1]
        return folder, name
    
    def fileGeneric(self, title:str) -> str:
        folder,name = self.exportName()
        fn = os.path.join(folder, title+'_'+name+'.csv')
        return fn
    
    def importGeneric(self, title:str, table:str, units:str) -> None:
        fmethod = getattr(self, title)
        fn = fmethod()
        if not os.path.exists(fn):
            return 1
        try:
            ret = plainIm(fn, ic=0)
            setattr(self, table, ret[0])
            setattr(self, units, ret[1])
        except:
            return 1
        else:
            return 0
        
    def exportGeneric(self, title:str, table:pd.DataFrame, units:dict, overwrite:bool=False) -> None:
        fmethod = getattr(self, title)
        fn = fmethod()
        if os.path.exists(fn) and not overwrite:
            return
        col = pd.MultiIndex.from_tuples([(k,v) for k, v in units.items()])
        data = np.array(table)
        df = pd.DataFrame(data, columns=col)
        df.to_csv(fn)
        logging.info(f'Exported {fn}')
        
    def fileSummary(self) -> str:
        '''name of the summary file'''
        return self.fileGeneric('summary')
    
    def importSummary(self) -> int:
        '''import an existing summary file. return 0 if successfully imported'''
        return self.importGeneric('fileSummary', 'summary', 'summaryUnits')
    
    def exportSummary(self, overwrite=False) -> None:
        return self.exportGeneric('fileSummary', self.summary, self.summaryUnits, overwrite=overwrite)
        
    def fileRelax(self) -> str:
        '''name of the relaxation file'''
        return self.fileGeneric('relax')
    
    def importRelax(self) -> int:
        '''import an existing relaxation file. return 0 if successfully imported'''
        return self.importGeneric('fileRelax', 'relaxation', 'relaxationUnits')
    
    def exportRelax(self, overwrite=False) -> None:
        return self.exportGeneric('fileRelax', self.relaxation, self.relaxationUnits)
        
    def fileDroplets(self) -> str:
        '''name of the droplets file'''
        return self.fileGeneric('droplets')
    
    def importDroplets(self) -> int:
        '''import an existing droplet list file. return 0 if successfully imported'''
        fn = self.fileDroplets()
        if not os.path.exists(fn):
            return 1
        try:
            self.dTabs.relabeledDroplets, self.dTabs.dropletTabUnits = plainIm(fn, ic=0)
            self.dTabs.dropletTab = self.dTabs.relabeledDroplets.copy()
            self.dTabs.finalFrame = max(self.dTabs.dropletTab.frame)
        except:
            return 1
        return 0
        
    def exportDroplets(self, overwrite=False) -> None:
        return self.exportGeneric('fileDroplets', self.dTabs.relabeledDroplets, self.dTabs.dropletTabUnits)

        
    def fileSigma(self) -> str:
        return self.fileGeneric('sigma')
    
    def importSigma(self) -> int:
        fn = self.fileSigma()
        if not os.path.exists(fn):
            return 1
        try:
            sigma, self.sigmaUnits = plainIm(fn, ic=0)
            self.fits = sigma.transpose().to_dict()
        except:
            return 1
        return 0
        
    def exportSigma(self, overwrite=False) -> None:
        fn = self.fileSigma()
        if os.path.exists(fn) and not overwrite:
            return
        df = pd.DataFrame(self.fits)
        df = df.transpose()
        col = pd.MultiIndex.from_tuples([(k,v) for k, v in self.sigmaUnits.items()])
        df = pd.DataFrame(np.array(df), columns=col)
        df.to_csv(fn)
        logging.info(f'Exported {fn}')
                    
    def fileFitPlot(self) -> str:
        folder,name = self.exportName()
        fn = os.path.join(folder, f'sigmaFits_{name}.png')
        return fn
            
    def exportFitPlot(self, overwrite:bool=False):
        fn = self.fileFitPlot()
        if os.path.exists(fn) and not overwrite:
            return
        self.summaryPlot.fig.savefig(fn, bbox_inches='tight', dpi=300)
        logging.info(f'Exported {fn}')
        
    def fileDropletPlot(self) -> str:
        folder,name = self.exportName()
        fn = os.path.join(folder, f'dropletsPlot_{name}.png')
        return fn
        
    def exportDropletPlot(self, overwrite:bool=False):
        fn = self.fileDropletPlot()
        if os.path.exists(fn) and not overwrite:
            return
        fig = self.plotDroplets(True, 'frame')
        fig.savefig(fn, bbox_inches='tight', dpi=300)
        logging.info(f'Exported {fn}')
    
    def exportAll(self, overwrite:bool=False) -> None:
        '''export all tables'''
        self.exportSummary(overwrite=overwrite)
        self.exportDroplets(overwrite=overwrite)
        self.exportSigma(overwrite=overwrite)
        self.exportFitPlot(overwrite=overwrite)
        self.exportDropletPlot(overwrite=overwrite)
        self.exportRelax(overwrite=overwrite)
        
        
    #----------------------------------
    
    def analyze(self, plot:bool=True) -> None:
        '''analyze the entire video. If we already started this, pick up where we left off.'''
        ret = self.importDroplets()
        if ret>0:
            # if no existing file, generate data
            if len(self.dTabs.dropletTab)==0:
                startFrame = 0
            else:
                startFrame = self.dTabs.finalFrame
            try:
                self.dTabs.readFrames(startFrame=startFrame)
            except Exception as e:
                logging.error(f'Error during readFrames: {e}')
                return
            try:
                logging.info('Consolidating droplets')
                self.dTabs.consolidateDroplets()
            except Exception as e:
                logging.error(f'Error during consolidateDroplets: {e}')
                return
        try:
            logging.info('Determining baseline')
            self.dTabs.baselineDroplets()
        except Exception as e:
            logging.error(f'Error during baselineDroplets: {e}')
            return
        try:
            logging.info('Splitting times')
            self.dTabs.splitTimes(self.profile)
        except Exception as e:
            logging.error(f'Error during splitTimes: {e}')
            return
        ret = self.importSummary()
        if ret>0:
            try:
                logging.info('Summarizing droplets')
                self.summarizeDroplets()
            except Exception as e:
                logging.error(f'Error during summarizeDroplets: {e}')
                traceback.print_exc()
                return
        plotfits = not(os.path.exists(self.fileFitPlot()))
        if plotfits:
            try:
                self.plotSummaries(True,label=self.droplet.name+'_'+self.matrix.name)
            except Exception as e:
                logging.error(f'Error during plotSummaries: {e}')
                return
        ret = self.importSigma()
        if ret>0:
            try:
                logging.info('Getting sigma')
                self.getSigma(plot=plotfits)
            except Exception as e:
                logging.error(f'Error during getSigma: {e}')
                return
        ret = self.importRelax()
        if ret>0:
            try:
                logging.info('Getting sigma from relaxation')
                self.getRelaxation()
            except Exception as e:
                logging.error(f'Error during relaxation: {e}')
                traceback.print_exc()
                return
        self.exportAll()