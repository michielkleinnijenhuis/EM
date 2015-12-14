#!/usr/bin/env python

import os
import sys
import argparse
from skimage import segmentation
import h5py

def main(argv):
    
    parser = argparse.ArgumentParser(description='Juggle around stacks.')
    
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
    
    
    args = parser.parse_args()  # shouldnt argv be an argument here?
    
    inputfile = args.inputfile
    outputfile = args.outputfile
    
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
#     element_size_um = [0.05, 0.0073, 0.0073] #[6.85,1,1] #[1,1,1]
    
    supervoxelsize = args.supervoxelsize
    compactness = args.compactness
    sigma = args.sigma
    
    n_segm = inds.size / supervoxelsize
    spac = [es for es in element_size_um]
    
    segments = segmentation.slic(inds, 
                                 n_segments=n_segm, 
                                 compactness=compactness, 
                                 sigma=sigma, 
                                 spacing=spac, 
                                 multichannel=mc, 
                                 convert2lab=False, 
                                 enforce_connectivity=False)
    segments = segments + 1
    sv = h5py.File(outputfile, 'w')
    sv.create_dataset(outfield, data=segments)
    if element_size_um is not None:
        sv[outfield].attrs['element_size_um'] = element_size_um
    sv.close()

if __name__ == "__main__":
    main(sys.argv[1:])

