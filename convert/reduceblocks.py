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
    parser.add_argument('inputstack', nargs=2,
                        help='...')
    parser.add_argument('-d', '--blockreduce', nargs=3, type=int, default=[1,7,7],
                        help='...')
    parser.add_argument('-f', '--func', default='np.amax',
                        help='...')
    parser.add_argument('-o', '--outpf', default='_br',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    inputstack = args.inputstack
    blockreduce = args.blockreduce
    func = args.func
    outpf = args.outpf

    stack, elsize, al = loadh5(datadir, dset_name + inputstack[0],
                               fieldname=inputstack[1])

    stack = block_reduce(stack, block_size=tuple(blockreduce), func=eval(func))
    elsize = [e*b for e, b in zip(elsize, blockreduce)]

    writeh5(stack, datadir, dset_name + outpf + inputstack[0],
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