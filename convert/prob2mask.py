#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.morphology import remove_small_objects, binary_dilation, ball
from skimage.measure import block_reduce

def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-p', '--probs',
                        default=['_probs', 'volume/predictions'], nargs=2,
                        help='...')
    parser.add_argument('-c', '--channel', type=int, default=None,
                        help='...')
    parser.add_argument('-b', '--blockreduce', nargs=3, type=int, default=None,
                        help='...')
    parser.add_argument('-l', '--lower_threshold', type=float, default=0,
                        help='...')
    parser.add_argument('-u', '--upper_threshold', type=float, default=1,
                        help='...')
    parser.add_argument('-s', '--size', type=int, default=0,
                        help='...')
    parser.add_argument('-d', '--dilation', type=int, default=0,
                        help='...')
    parser.add_argument('-o', '--outpf', default='_mask',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    probs = args.probs
    channel = args.channel
    blockreduce = args.blockreduce
    lower_threshold = args.lower_threshold
    upper_threshold = args.upper_threshold
    size = args.size
    dilation = args.dilation
    outpf = args.outpf

    prob, elsize, al = loadh5(datadir, dset_name + probs[0],
                              fieldname=probs[1], channel=channel)

    mask = np.logical_and(prob > lower_threshold, prob <= upper_threshold)
    if size:
        remove_small_objects(mask, min_size=size, in_place=True)
    if dilation:
        mask = binary_dilation(mask, selem=ball(dilation))

    if blockreduce:
        mask = block_reduce(mask, block_size=tuple(blockreduce), func=np.amax)
        elsize = [e*b for e, b in zip(elsize, blockreduce)]

    writeh5(mask, datadir, dset_name + outpf, dtype='uint8',
            element_size_um=elsize[:3], axislabels=al[:3])


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
    main(sys.argv[1:])
