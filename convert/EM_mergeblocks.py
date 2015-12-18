#!/usr/bin/env python

"""
python EM_mergeblocks.py ...
"""

import os
import sys
from argparse import ArgumentParser
import h5py

def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('outputfile', help='...')
    parser.add_argument('field', help='...')
    parser.add_argument('layout', help='...')
    parser.add_argument('-d', '--datatype', default='float32', help='...')
    parser.add_argument('-s', '--shape', nargs=4, type=int, default=[100,4111,4235,6], help='...')
    parser.add_argument('-c', '--chunksize', nargs=4, type=int, default=[20,20,20,6], help='...')
    parser.add_argument('-e', '--element_size_um', nargs=4, type=float, default=[0.05,0.0073,0.0073,1], help='...')
    parser.add_argument('-f', '--files', nargs='*', help='...')
    
    args = parser.parse_args()
    
    outputfile = args.outputfile
    field = args.field
    layout = args.layout
    datatype = args.datatype
    shape = args.shape
    chunksize = args.chunksize
    element_size_um = args.element_size_um
    files = args.files
#     datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU'
#     datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
#     dataset = 'm000'
#     outputfile = dataset + '_probs.h5'
#     field = '/volume/predictions'
#     layout = 'zyxc'
#     chunksize = [20,20,20,6]
#     datatype = 'float32'
#     element_size_um = f['stack'].attrs['element_size_um']
    #shape = f['stack'].shape
    #shape.append(6)
#     shape = (100, 4111, 4235, 6)
#     shape = (460, 4111, 4235, 6)
    
    for i, filename in enumerate(files):
        
        f = h5py.File(filename, 'r')
        _, tail = os.path.split(filename)
        parts = tail.split("_")
        x = int(parts[1].split("-")[0])
        X = int(parts[1].split("-")[1])
        y = int(parts[2].split("-")[0])
        Y = int(parts[2].split("-")[1])
        z = int(parts[3].split("-")[0])
        Z = int(parts[3].split("-")[1])
        
        if i == 0:
            otype = 'a' if os.path.isfile(outputfile) else 'w'
            g = h5py.File(outputfile, otype)
            outds = g.create_dataset(field, shape, 
                                     chunks=tuple(chunksize), 
                                     dtype=datatype)
            outds.attrs['element_size_um'] = element_size_um
            for i,l in enumerate(layout):
                outds.dims[i].label = l
        
        g[field][z:Z,y:Y,x:X,:] = f[field][:,:,:,:]
        f.close()
    
    g.close()


if __name__ == "__main__":
    main(sys.argv)
