#!/usr/bin/env python
'''Functions for handling files'''

# external packages
import os, sys
import re
import shutil
import time
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging
import pandas as pd
import numpy as np
import multiprocessing as mp

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from config import cfg
from vidRead import vidInfo, plainIm

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# info
__author__ = "Leanne Friedrich"
__copyright__ = "This data is publicly available according to the NIST statements of copyright, fair use and licensing; see https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software"
__credits__ = ["Leanne Friedrich"]
__license__ = "MIT"
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
        elif s[0:5]=='PDMSS':
            # mineral oil based fluid
            self.base='PDMS_3_silicone_25'
            self.rheModifier = 'fumed silica'
            s = s[5:] # remove M from name
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
            if s=='water':
                s = 0
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
            raise ValueError('No rheology fit found')
        if len(entry)>1:
            print(entry)
            logging.error(f'Multiple rheology fits found for fluid {self.name}')
            raise ValueError('Multiple rheology fits found')
        entry = entry.iloc[0]
        self.tau0 = entry['y2_tau0'] # Pa
        self.k = entry['y2_k'] 
        self.n = entry['y2_n']
        self.eta0 = entry['y2_eta0'] # these values are for Pa.s, vs. frequency in 1/s
        return
        
    def visc(self, gdot:float) -> float:
        '''get the viscosity of the fluid in Pa*s at shear rate gdot in Hz'''
        if gdot==0:
            return self.eta0
        if self.k==0:
            return self.eta0
        mu = self.k*(abs(gdot)**(self.n-1)) + self.tau0/(abs(gdot))
#         mu = min(mu, self.eta0)
        return mu
        
            
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
            spl = re.split(',', data)
            spl2 = []
            i=0
            while i<len(spl):
                s = spl[i]
                if s=='stdy':
                    spl2.append(spl[i:i+4]+spl[i+5:i+8])
                    i = i+8
                elif s=='relax':
                    spl2.append(spl[i:i+3]+[0]+spl[i+4:i+7])
                    i = i+7
                else:
                    i = i+1
            cols = ['mode', 'gap', 'strain', 'rate', 'freq', 'direction', 'time' ]
            self.units = {'mode':'', 'gap':'um', 'strain':'', 'rate':'1/s','freq':'','direction':'','time':'s'}
            self.table = pd.DataFrame(spl2, columns=cols)

class Test:
    '''stores info about a test for a single material combo and shear profile'''
    
    def __init__(self, folder:str):
        if not os.path.isdir(folder):
            raise NameError('Input to Test must be a folder name')
        self.folder = folder
        bn = os.path.basename(folder)
        spl = re.split('_', bn)
        if not len(spl)==3:
            raise ValueError(f'Unknown naming format for {folder}. getFluids assumes format droplet_matrix_profile')
        droplet = spl[0]
        matrix = spl[1]
        profile = spl[2]
    
        try:
            self.droplet = Fluid(droplet)
            self.matrix = Fluid(matrix)
        except Exception as e:
            raise ValueError(f'Failed to initialize fluids: {e}.')
        pfile = os.path.join(os.path.dirname(folder), profile+'.mot') # profile should be in parent folder
        if not os.path.exists(pfile):
            raise NameError(f'profile {pfile} not found')
        self.profile = Profile(pfile)
        
        self.videos = [os.path.join(folder, s) for s in os.listdir(folder)] # list of videos
        self.videos = list(filter(lambda x:'.mp4' in x, self.videos))
        
    def prnt(self):
        print('Droplet:',self.droplet.__dict__)
        print('Matrix:',self.matrix.__dict__)
        print('Profile:\n', self.profile.table)
        print('Videos:', [os.path.basename(s) for s in self.videos])   
        
    def done(self) -> bool:
        '''determine if all files are done'''
        for i in range(len(self.videos)):
            vi = vidInfo(self, i)
            if not vi.done():
                return False
        return True
        
    def analyze(self) -> None:
        '''analyze all of the videos in the folder'''
        if self.done():
            return
        self.prnt()
        for i in range(len(self.videos)):
            vi = vidInfo(self, i)
            vi.analyze()
        self.sigmaSummary()
        self.importRelax()
            
    def importSummaries(self) -> None:
        '''import summaries'''
        self.summaries = []
        for i in range(len(self.videos)):
            vi = vidInfo(self, i)
            ret = vi.importSummary()
            if ret==0:
                if len(self.summaries)==0 and len(vi.summary)>0:
                    self.summaries = vi.summary
                else:
                    if len(vi.summary)>0:
                        vi.summary.dropNum = vi.summary.dropNum+max(self.summaries.dropNum)+1
                        self.summaries = pd.concat([self.summaries, vi.summary])
                    
    def importRelax(self) -> None:
        self.relax = []
        for i in range(len(self.videos)):
            vi = vidInfo(self, i)
            ret = vi.importRelax()
            if ret==0:
                if len(self.relax)==0 and len(vi.relaxation)>0:
                    self.relax = vi.relaxation
                else:
                    if len(vi.relaxation)>0:
                        vi.relaxation.dropNum = vi.relaxation.dropNum+max(self.relax.dropNum)+1
                        self.relax = pd.concat([self.relax, vi.relaxation])
        if len(self.relax)>0:
            vi.file = 'all_'
            vi.relaxation = self.relax
            vi.exportRelax(overwrite=True)
                    
    def sigmaSummary(self, tcrit:float=0.25, plot:bool=True) -> None:
        '''import summaries and get sigma. tcrit is critical droplet size as a fraction of the gap width'''
        self.importSummaries()
        if len(self.summaries)==0:
            return
        gap = float(self.profile.table.loc[0, 'gap'])/10**6
        small = self.summaries[self.summaries.r0<gap*tcrit]
        if len(small)<5:
            return
        vi = vidInfo(self, 0) # create a dummy vidInfo object
        vi.file='all_'
        vi.summary = small
        if plot:
            vi.plotSummaries(True,label=self.droplet.name+'_'+self.matrix.name)
        vi.getSigma(plot=plot)
        vi.exportSigma(overwrite=True)
        vi.exportFitPlot(overwrite=True) 
        
        
def analyzeRecursive(folder:str) -> None:
    '''go through all folders in the folder and analyze recursively'''
    try:
        t = Test(folder)
    except:
        if not os.path.isdir(folder):
            return
        else:
            for f in os.listdir(folder):
                f1f = os.path.join(folder, f)
                analyzeRecursive(f1f)
    else:
        try:
            t.analyze()
        except Exception as e:
            logging.error('Error in '+folder)
            logging.error(str(e))
            
            
def makeTestList(folder:str) -> None:
    try:
        t = Test(folder)
    except:
        if not os.path.isdir(folder):
            return []
        else:
            tlist = []
            for f in os.listdir(folder):
                f1f = os.path.join(folder, f)
                tlist = tlist + makeTestList(f1f)
            return tlist
    else:
        return [t]
    
    
def tanalyze(t:Test) -> None:
    print(t.folder)
#     t.analyze()

def analyzeParallel(folder:str) -> None:
    '''analyze folders in parallel'''
    tlist = makeTestList(folder)
    pool = mp.Pool(mp.cpu_count())
    pool.map_async(tanalyze, tlist)
    pool.close()
    
    
def combineSummaries(folder:str) -> None:
    sumall = []
    for f in os.listdir(folder):
        f1 = os.path.join(folder, f)
        if os.path.isdir(f1):
            r = combineSummaries(f1)
            if len(r)>0:
                if len(sumall)>0:
                    sumall = pd.concat([sumall, r])
                else:
                    sumall = r
        else:
            if 'summary' in f1:
                r, d = plainIm(f1, 0)
                if len(r)>0:
                    times = re.split('_', os.path.basename(folder))
                    name = times[-1][-3:]
                    r['vid'] = [name for i in range(len(r))]
                    r['dropNum'] = [row['vid']+'.'+str(int(row['dropNum'])) for i,row in r.iterrows()]
                    if len(sumall)==0:
                        sumall = r
                        di = d
                    else:
                        sumall = pd.concat([sumall, r])
            else:
                sumall = []
    return sumall



def combineRelax(folder:str) -> None:
    sumall = []
    for f in os.listdir(folder):
        f1 = os.path.join(folder, f)
        if os.path.isdir(f1):
            r = combineRelax(f1)
            if len(r)>0:
                if len(sumall)>0:
                    sumall = pd.concat([sumall, r])
                else:
                    sumall = r
        else:
            if 'relax_' in f1:
                r, d = plainIm(f1, 0)
                if len(r)>0:
                    times = re.split('_', os.path.basename(folder))
                    name = times[-1][-3:]
                    r['vid'] = [name for i in range(len(r))]
                    r['dropNum'] = [row['vid']+'.'+str(int(row['dropNum'])) for i,row in r.iterrows()]
                    if len(sumall)==0:
                        sumall = r
                        di = d
                    else:
                        sumall = pd.concat([sumall, r])
            else:
                sumall = []
    return sumall
        

        
    
    
    
            
        
    