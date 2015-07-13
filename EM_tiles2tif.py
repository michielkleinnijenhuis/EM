#!/usr/bin/env python

"""
Convert DM3 montage to tifs and rename to <[slc]_m[mon].tif>.
"""

import os
import sys
from os import path
import glob
from loci.plugins import BF
from ij import IJ

def main(argv):
    
    inputdir = 'INPUTDIR'
    outputdir = 'OUTPUTDIR'
    montage = 'MONTAGE'
    
    # Get list of DM3 files
    infiles = glob.glob(path.join(inputdir, '*.dm3'))
    # TODO: implement x,y,z selections
    
    for infile in infiles:
        imp = BF.openImagePlus(infile)
        IJ.save(imp[0], path.join(outputdir, infile[-8:-4] + '_m' + montage + '.tif'))

if __name__ == "__main__":
   main(sys.argv[1:])
