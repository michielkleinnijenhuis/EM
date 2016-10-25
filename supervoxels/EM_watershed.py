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
    parser.add_argument('--maskDS', default=['_maskDS', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('-p', '--probs',
                        default=['_probs', 'volume/predictions'], nargs=2,
                        help='...')
    parser.add_argument('-c', '--channel', type=int, default=None,
                        help='...')
    parser.add_argument('-l', '--lower_threshold', type=float, default=0,
                        help='...')
    parser.add_argument('-u', '--upper_threshold', type=float, default=1,
                        help='...')
    parser.add_argument('-s', '--seed_size', type=int, default=64,
                        help='...')
    parser.add_argument('-o', '--outpf', default='_ws',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    maskDS = args.maskDS
    maskMM = args.maskMM
    probs = args.probs
    channel = args.channel

    lower_threshold = args.lower_threshold
    upper_threshold = args.upper_threshold
    seed_size = args.seed_size
    outpf = args.outpf

    maskDS, elsize, al = loadh5(datadir, dset_name + maskDS[0],
                                fieldname=maskDS[1], dtype='bool')
    maskMM = loadh5(datadir, dset_name + maskMM[0],
                    fieldname=maskMM[1], dtype='bool')[0]
    prob = loadh5(datadir, dset_name + probs[0],
                  fieldname=probs[1], channel=channel)[0]

    seeds = label(np.logical_and(prob > lower_threshold,
                                 prob <= upper_threshold))[0]
    remove_small_objects(seeds, min_size=seed_size, in_place=True)
    seeds = relabel_sequential(seeds)[0]

    MA = watershed(-prob, seeds, mask=np.logical_and(~maskMM, maskDS))

    writeh5(MA, datadir, dset_name + outpf, dtype='int32',
            element_size_um=elsize, axislabels=al)


# ========================================================================== #
# function defs
# ========================================================================== #


def loadh5(datadir, dname, fieldname='stack', dtype=None, channel=None):
    """"""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        if channel is not None:
            stack = f[fieldname][:, :, :, channel]
        else:
            stack = f[fieldname][:, :, :, :]

    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    if 'DIMENSION_LABELS' in f[fieldname].attrs.keys():
        axislabels = [d.label for d in f[fieldname].dims]
    else:
        axislabels = None

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """"""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    if element_size_um is not None:
        g[fieldname].attrs['element_size_um'] = element_size_um
    if axislabels is not None:
        for i, l in enumerate(axislabels):
            g[fieldname].dims[i].label = l

    g.close()


if __name__ == "__main__":
    main(sys.argv[1:]
)
