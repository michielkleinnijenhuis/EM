#!/usr/bin/env python

"""
Convert DM3 montage to tifs.
"""

import sys
from os import path
import glob
from loci.plugins import BF
from ij import IJ

def main(argv):
    
    inputdir = 'INPUTDIR'
    outputdir = 'OUTPUTDIR'
    output_postfix = 'OUTPUT_POSTFIX'
    
    infiles = glob.glob(path.join(inputdir, '*.dm3'))
    
    for infile in infiles:
        imp = BF.openImagePlus(infile)
        head, tail = path.split(infile)
        filename, ext = path.splitext(tail)
        IJ.save(imp[0], path.join(outputdir, filename + output_postfix + ext))

if __name__ == "__main__":
    main(sys.argv[1:])
