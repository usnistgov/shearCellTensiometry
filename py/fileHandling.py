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

# local packages
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from config import cfg

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
            print(s)
            self.rheWt = ''
        else:
            self.rheWt = wt
            
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
        print('Profile:', self.profile.table)
        print('Videos:', [os.path.basename(s) for s in self.videos])
        
    
    
    
            
        
    