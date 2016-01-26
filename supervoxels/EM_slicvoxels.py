#!/usr/bin/env python

import os
import sys
import argparse
from skimage import segmentation
import h5py
import numpy as np

def main(argv):
    
    parser = argparse.ArgumentParser(description='...')
    
    parser.add_argument('inputfile', help='the inputfile')
    parser.add_argument('outputfile', help='the outputfile')
    parser.add_argument('-f', '--fieldnamein', 
                        help='input hdf5 fieldname <stack>')
    parser.add_argument('-g', '--fieldnameout', 
                        help='output hdf5 fieldname <stack>')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*', 
                        help='dataset element sizes (in order of outlayout)')
    parser.add_argument('-s', '--supervoxelsize', type=int, default=500, help='...')
    parser.add_argument('-c', '--compactness', type=float, default=0.2, help='...')
    parser.add_argument('-o', '--sigma', type=float, default=1, help='...')
    parser.add_argument('-u', '--enforceconnectivity', action='store_true', help='...')
    
    
    args = parser.parse_args()  # shouldnt argv be an argument here?
    
    inputfile = args.inputfile
    outputfile = args.outputfile
    sigma = args.sigma
    
    f = h5py.File(inputfile, 'r')
    
    if args.fieldnamein:
        infield = args.fieldnamein
    else:
        grps = [name for name in f]
        infield = grps[0]
    
    if args.fieldnameout:
        outfield = args.fieldnameout
    elif inputfile != outputfile:
        outfield = infield
    else:
        outfield = 'stack'
    
    mc = True if len(f[infield].dims) == 4 else False
    if mc:
        inds = f[infield][:,:,:,:]
    else:
        inds = f[infield][:,:,:]
    
    if args.element_size_um:
        element_size_um = args.element_size_um
    elif 'element_size_um' in f[infield].attrs.keys():
        element_size_um = f[infield].attrs['element_size_um']
    else:
        element_size_um = None
    
    n_segm = inds.size / args.supervoxelsize
    compactness = args.compactness
    spac = [es for es in np.absolute(element_size_um)]
    ec = args.enforceconnectivity
    
    segments = segmentation.slic(inds, 
                                 n_segments=n_segm, 
                                 compactness=compactness, 
                                 spacing=spac, 
                                 sigma=sigma, 
                                 multichannel=mc, 
                                 convert2lab=False, 
                                 enforce_connectivity=ec)
    segments = segments + 1
    print("Number of supervoxels: ", np.amax(segments))
    sv = h5py.File(outputfile, 'w')
    dset = sv.create_dataset(outfield, segments.shape, 
                             dtype='int64', compression='gzip')
    dset[:] = segments
    if element_size_um is not None:
        sv[outfield].attrs['element_size_um'] = element_size_um
    sv.close()

if __name__ == "__main__":
    main(sys.argv[1:])

