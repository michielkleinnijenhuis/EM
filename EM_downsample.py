#!/usr/bin/env python

"""
Downsample a range of slices.
"""

import sys
from os import path, makedirs
import glob
from argparse import ArgumentParser

from skimage import io
from skimage.transform import resize
from scipy.misc import imsave

import numpy as np
from mpi4py import MPI

def main(argv):
    """Downsample a series of images."""
    
    parser = ArgumentParser(description='Downsample images.')
    parser.add_argument('inputdir', help='a directory with images')
    parser.add_argument('outputdir', help='the output directory')
    parser.add_argument('-n', '--nzfills', type=int, default=4, 
                        help='the number of characters at the end that define z')
    parser.add_argument('-r', '--regex', default='*.tif', 
                        help='regular expression to select files with')
    parser.add_argument('-d', '--ds_factor', type=int, default=4, 
                        help='the factor to downsample the images by')
    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')
    args = parser.parse_args()
    
    inputdir = args.inputdir
    outputdir = args.outputdir
#     if not path.exists(outputdir):
#         makedirs(outputdir)
    
    nzfills = args.nzfills
    regex = args.regex
    files = glob.glob(path.join(inputdir, regex))
#     (root, ext) = path.splitext(files[0])
#     (head, tail) = path.split(root)
#     prefix = tail[:-nzfills]
    
    firstimage = io.imread(files[0])
    x = args.x
    X = args.X if args.X else firstimage.shape[1]
    y = args.y
    Y = args.Y if args.Y else firstimage.shape[0]
    z = args.z
    Z = args.Z if args.Z else len(files)
    
    ds_factor = args.ds_factor
    
    usempi = args.usempi
    
    n_slcs = Z-z
    slcnrs = np.array(range(0, n_slcs), dtype=int)
    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
         
        # scatter the slices
        local_nslcs = np.ones(size, dtype=int) * n_slcs / size
        local_nslcs[0:n_slcs % size] += 1
        local_slcnrs = np.zeros(local_nslcs[rank], dtype=int)
        displacements = tuple(sum(local_nslcs[0:r]) for r in range(0, size))
        comm.Scatterv([slcnrs, tuple(local_nslcs), displacements, 
                       MPI.SIGNED_LONG_LONG], local_slcnrs, root=0)
    else:
        local_slcnrs = slcnrs
    
    for slc in local_slcnrs:
        original = io.imread(path.join(inputdir, str(slc).zfill(nzfills) + '.tif'))
        sub = original[x:X,y:Y]
        ds_sub = resize(sub, (sub.shape[0] / ds_factor, 
                                   sub.shape[1] / ds_factor))
        imsave(path.join(outputdir, str(slc).zfill(nzfills) + '.tif'), ds_sub)

if __name__ == "__main__":
    main(sys.argv[1:])
