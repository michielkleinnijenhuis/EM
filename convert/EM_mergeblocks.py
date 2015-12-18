#!/usr/bin/env python

"""
python EM_mergeblocks.py ...
"""

import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np

def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('-i', '--inputfiles', nargs='*', help='...')
    parser.add_argument('-f', '--field', default='stack', help='...')
    parser.add_argument('-o', '--outputfile', help='...')
    parser.add_argument('-l', '--outlayout', help='...')
    parser.add_argument('-c', '--chunksize', nargs='*', type=int, help='...')
    parser.add_argument('-e', '--element_size_um', nargs='*', type=float, help='...')
    
    args = parser.parse_args()
    
    inputfiles = args.inputfiles
    outputfile = args.outputfile
    field = args.field
    chunksize = args.chunksize
    element_size_um = args.element_size_um
    outlayout = args.outlayout
    
    for i, filename in enumerate(inputfiles):
        
        f = h5py.File(filename, 'r')
        head, tail = os.path.split(filename)
        parts = tail.split("_")
        x = int(parts[1].split("-")[0])
        X = int(parts[1].split("-")[1])
        y = int(parts[2].split("-")[0])
        Y = int(parts[2].split("-")[1])
        z = int(parts[3].split("-")[0])
        Z = int(parts[3].split("-")[1])
        
        if i == 0:
            
            if not outputfile:
                outputfile = os.path.join(head, parts[0] + '_' + parts[-1])
            
            if not chunksize:
                try:
                    chunksize = f[field].chunks
                except:
                    pass
            
            ndims = len(f[field].shape)
            maxshape = [None] * ndims
            
            otype = 'a' if os.path.isfile(outputfile) else 'w'
            g = h5py.File(outputfile, otype)
            outds = g.create_dataset(field, f[field].shape, 
                                     chunks=chunksize, 
                                     dtype=f[field].dtype, 
                                     maxshape=maxshape)
            
            if element_size_um:
                outds.attrs['element_size_um'] = element_size_um
            else:
                try:
                    outds.attrs['element_size_um'] = f[field].attrs['element_size_um']
                except:
                    pass
            
            if outlayout:
                for i,l in enumerate(outlayout):
                    outds.dims[i].label = l
            else:
                try:
                    for i,d in enumerate(f[field].dims):
                        outds.dims[i].label = d.label
                except:
                    pass
        
        for i, newmax in enumerate([Z,Y,X]):
            if newmax > g[field].shape[i]:
                g[field].resize(newmax, i)
        
        if ndims == 3:
            g[field][z:Z,y:Y,x:X] = f[field][:,:,:]
        elif ndims == 4:
            g[field][z:Z,y:Y,x:X,:] = f[field][:,:,:,:]
        
        f.close()
    
    g.close()


if __name__ == "__main__":
    main(sys.argv)
