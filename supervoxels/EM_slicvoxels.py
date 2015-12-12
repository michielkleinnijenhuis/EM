#!/usr/bin/env python

import os
import sys
import getopt
from skimage import segmentation
import h5py

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:f:g:s:",
                                   ["inputfile=","outputfile=",
                                    "fieldname_in","fieldname_out",
                                    "supervoxelsize"])
    except getopt.GetoptError:
        print 'EM_slicvoxels.py -i <inputfile> -o <outputfile> -f <fieldname hdf5 input> -g <fieldname hdf5 output> -s <supervoxelsize>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'EM_slicvoxels.py -i <inputfile> -o <outputfile> -f <fieldname hdf5 input> -g <fieldname hdf5 output> -s <supervoxelsize>'
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            inputfile = arg
        elif opt in ("-o", "--outputfile"):
            outputfile = arg
        elif opt in ("-f", "--fieldname_in"):
            fieldname_in = arg
        elif opt in ("-g", "--fieldname_out"):
            fieldname_out = arg
        elif opt in ("-s", "--supervoxelsize"):
            supervoxelsize = int(arg)
    
    f = h5py.File(inputfile, 'r')
    
    mc = True if len(f[fieldname_in].dims) == 4 else False
    
    if mc:
        inds = f[fieldname_in][:,:,:,:]
    else:
        inds = f[fieldname_in][:,:,:]
    
#     supervoxelsize = 500
    n_segm = inds.size / supervoxelsize
    comp = 0.1  # TODO make into argument
    spac = [6.85,1,1] #     spac = [1,1,1]  # TODO read spacing from hdf5  [0.05,0.0073,0.0073]???
    sig = 1
    
    
    segments = segmentation.slic(inds, 
                                 n_segments=n_segm, 
                                 compactness=comp, 
                                 sigma=sig, 
                                 spacing=spac, 
                                 multichannel=mc, 
                                 convert2lab=False, 
                                 enforce_connectivity=False)
    segments = segments + 1
    sv = h5py.File(outputfile, 'w')
    sv.create_dataset(fieldname_out, data=segments)
    sv.close()

if __name__ == "__main__":
   main(sys.argv[1:])

