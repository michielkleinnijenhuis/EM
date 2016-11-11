#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.measure import regionprops

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


# TODO: write elsize and axislabels
def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume_ws', default=['', 'stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-L', '--labelvolume_2D', default=['', 'stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['', 'stack'],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    outpf = args.outpf
    labelvolume_ws = args.labelvolume_ws
    labelvolume_2D = args.labelvolume_2D

    # open datasets
    lname = dset_name + labelvolume_ws[0] + '.h5'
    lpath = os.path.join(datadir, lname)
    lfile = h5py.File(lpath, 'r')
    lstack = lfile[labelvolume_ws[1]]
    labels_ws = lstack[:,:,:]
    elsize, al = get_h5_attributes(lstack)
    lfile.close()

    lname = dset_name + labelvolume_2D[0] + '.h5'
    lpath = os.path.join(datadir, lname)
    lfile = h5py.File(lpath, 'r')
    lstack = lfile[labelvolume_2D[1]]
    labels_2D = lstack[:,:,:]
    lfile.close()

    labelsets = {i: set(np.unique(labels_2D[i,:,:]))
                 for i in range(labels_2D.shape[0])}

    ulabels = np.unique(labels_ws)
    m = {l: np.array([True if l in lsv else False 
                      for _, lsv in labelsets.items()])
         for l in ulabels}

    rp = regionprops(labels_ws)
    for prop in rp:
        print(prop.label)
        z, y, x, Z, Y, X = tuple(prop.bbox)
        imregion = labels_ws[z:Z,y:Y,x:X]
        mask = prop.image
        mask[m[prop.label][z:Z],:,:] = 0
        imregion[mask] = 0
    
    gname = dset_name + outpf[0]
    gpath = os.path.join(datadir, gname + '.h5')
    gfile = h5py.File(gpath, 'w')
    outds = gfile.create_dataset(outpf[1], labels_ws.shape,
                                 dtype='uint32', compression='gzip')
    write_h5_attributes(gfile[outpf[1]], elsize, al)
    outds[:,:,:] = labels_ws
    gfile.close()

# ========================================================================== #
# function defs
# ========================================================================== #


def get_h5_attributes(stack):
    """Get attributes from a stack."""

    element_size_um = axislabels = None

    if 'element_size_um' in stack.attrs.keys():
        element_size_um = stack.attrs['element_size_um']

    if 'DIMENSION_LABELS' in stack.attrs.keys():
        axislabels = stack.attrs['DIMENSION_LABELS']

    return element_size_um, axislabels


def write_h5_attributes(stack, element_size_um=None, axislabels=None):
    """Write attributes to a stack."""

    if element_size_um is not None:
        stack.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, l in enumerate(axislabels):
            stack.dims[i].label = l


if __name__ == "__main__":
    main(sys.argv[1:])
