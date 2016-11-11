#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from scipy.ndimage import label
# from skimage.measure import label
from skimage.segmentation import relabel_sequential
from skimage.morphology import watershed, remove_small_objects


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-p', '--probs', nargs=2,
                        default=['_probs', 'volume/predictions'],
                        help='...')
    parser.add_argument('-c', '--channel', type=int, default=None,
                        help='...')
    parser.add_argument('-D', '--maskDS', nargs=2, default=['_maskDS', 'stack'],
                        help='...')
    parser.add_argument('-M', '--maskMM', nargs=2, default=['_maskMM', 'stack'],
                        help='...')
    parser.add_argument('-S', '--seedimage', nargs=2, default=None,
                        help='...')
    parser.add_argument('-l', '--lower_threshold', type=float, default=0,
                        help='...')
    parser.add_argument('-u', '--upper_threshold', type=float, default=1,
                        help='...')
    parser.add_argument('-s', '--seed_size', type=int, default=64,
                        help='...')
    parser.add_argument('-r', '--relabel', action='store_true',
                        help='...')
    parser.add_argument('-q', '--min_labelsize', type=int, default=None,
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2, default=['_ws', 'stack'],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    maskDS = args.maskDS
    maskMM = args.maskMM
    seedimage = args.seedimage
    probs = args.probs
    channel = args.channel

    lower_threshold = args.lower_threshold
    upper_threshold = args.upper_threshold
    seed_size = args.seed_size
    relabel = args.relabel
    min_labelsize = args.min_labelsize

    outpf = args.outpf

    gname = dset_name + outpf[0] + '.h5'
    gpath = os.path.join(datadir, gname)
    pname = dset_name + probs[0] + '.h5'
    ppath = os.path.join(datadir, pname)
    dsname = dset_name + maskDS[0] + '.h5'
    dspath = os.path.join(datadir, dsname)
    mmname = dset_name + maskMM[0] + '.h5'
    mmpath = os.path.join(datadir, mmname)

    g = h5py.File(gpath, 'w')
    p = h5py.File(ppath, 'r')
    pstack = p[probs[1]]
    ds = h5py.File(dspath, 'r')
    dstack = ds[maskDS[1]]
    mm = h5py.File(mmpath, 'r')
    mstack = mm[maskMM[1]]

    outds = g.create_dataset(outpf[1], mstack.shape,
                             dtype='uint32',
                             compression='gzip')
    elsize, al = get_h5_attributes(mstack)
    write_h5_attributes(g[outpf[1]], elsize, al)

    if seedimage is not None:
        print('labeling seeds')
        seeds = label(np.logical_and(pstack[:,:,:,channel] > lower_threshold,
                                     pstack[:,:,:,channel] <= upper_threshold))[0]
        remove_small_objects(seeds, min_size=seed_size, in_place=True)
    else:
        sname = dset_name + seedimage[0] + '.h5'
        spath = os.path.join(datadir, sname)
        s = h5py.File(spath, 'r')
        sstack = s[seedimage[1]]
        seeds = sstack[:,:,:]
        s.close()

    if relabel:
        seeds = relabel_sequential(seeds)[0]

    print('running watershed')
    MA = watershed(-pstack[:,:,:,channel], seeds,
                   mask=np.logical_and(~mstack[:,:,:], dstack[:,:,:]))

    if min_labelsize is not None:
        remove_small_objects(MA, min_size=min_labelsize, in_place=True)

    outds = MA

    g.close()
    p.close()
    ds.close()
    mm.close()


# ========================================================================== #
# function defs
# ========================================================================== #


def loadh5(datadir, dname, fieldname='stack', dtype=None):
    """Load a h5 stack."""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]

    element_size_um, axislabels = get_h5_attributes(f[fieldname])

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """Write a h5 stack."""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    write_h5_attributes(g[fieldname], element_size_um, axislabels)

    g.close()


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
    main(sys.argv[1:]
)
