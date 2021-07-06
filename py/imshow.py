#!/usr/bin/env python
'''Functions for reading and analyzing shear droplet videos'''

# external packages
from matplotlib import pyplot as plt
import cv2 as cv
from typing import List, Dict, Tuple, Union, Any, TextIO
import logging

# local packages

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

####### DISPLAY TOOLS

def imshow(*args, scale:float=10) -> None:
    '''displays cv image(s) in jupyter notebook using matplotlib'''
    if len(args)>1:
        f, axs = plt.subplots(1, len(args), figsize=(1*scale,len(args)*scale))
        for i, im in enumerate(args):
            if len(im.shape)>2:
                # color
                axs[i].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            else:
                # B&W
                axs[i].imshow(im, cmap='Greys')
    else:
        f, ax = plt.subplots(1, len(args), figsize=(1*scale,len(args)*scale))
        im = args[0]
        if len(im.shape)>2:
            ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        else:
            ax.imshow(im, cmap='Greys')