import os
import sys
sys.path.append("../")
from py.config import cfg
import py.fileHandling as fh
import py.logs as logs
LOGGERDEFINED = logs.openLog('analyzeScript.py', False, level='DEBUG', exportLog=True)

fh.analyzeRecursive(cfg.path.vids)