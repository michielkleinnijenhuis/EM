#!/usr/bin/env python

"""
Downsample a range of slices.
"""

import sys
from os import path
import getopt

from skimage import io
from skimage.transform import resize

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:d:x:X:y:Y:z:Z:",
                                   ["inputdir=","outputdir=","ds_factor",
                                    "x_start=","x_end",
                                    "y_start=","y_end",
                                    "z_start=","z_end"])
    except getopt.GetoptError:
        print 'EM_downsample.py -i <inputdir> -o <outputdir> \
        -d <downsample factor> \
        -x <x_start> -X <x_end> \
        -y <y_start> -Y <y_end> \
        -z <z_start> -Z <z_end>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'EM_downsample.py -i <inputdir> -o <outputdir> \
            -d <downsample factor> \
            -x <x_start> -X <x_end> \
            -y <y_start> -Y <y_end> \
            -z <z_start> -Z <z_end>'
            sys.exit()
        elif opt in ("-i", "--inputdir"):
            inputdir = arg
        elif opt in ("-o", "--outputdir"):
            outputdir = arg
        elif opt in ("-d", "--ds_factor"):
            ds_factor = int(arg)
        elif opt in ("-x", "--x_start"):
            x_start = int(arg)
        elif opt in ("-X", "--x_end"):
            x_end = int(arg)
        elif opt in ("-y", "--y_start"):
            y_start = int(arg)
        elif opt in ("-Y", "--y_end"):
            y_end = int(arg)
        elif opt in ("-z", "--z_start"):
            z_start = int(arg)
        elif opt in ("-Z", "--z_end"):
            z_end = int(arg)
    
    for slc in range(z_start, z_end):
        original = io.imread(path.join(inputdir, str(slc).zfill(4) + '.tif'))
        #sub = original[x_start:x_end,y_start:y_end]
        sub = original
        ds_sub = resize(sub, (sub.shape[0] / ds_factor, 
                                   sub.shape[1] / ds_factor))
        io.imsave(path.join(outputdir, str(slc).zfill(4) + '.tif'), ds_sub)

if __name__ == "__main__":
    main(sys.argv[1:])
