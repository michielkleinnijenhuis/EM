#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.measure import block_reduce
from scipy.stats.mstats import mode

def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('inputstack', nargs=2,
                        help='...')
    parser.add_argument('-d', '--blockreduce', nargs=3, type=int,
                        default=[1,7,7],
                        help='...')
    parser.add_argument('-f', '--func', default='np.amax',
                        help='...')
    parser.add_argument('-o', '--outpf', default='br',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    inputstack = args.inputstack
    blockreduce = args.blockreduce
    func = args.func
    outpf = args.outpf

    fname = os.path.join(datadir, dset_name + inputstack[0] + '.h5')
    f = h5py.File(fname, 'r')
    fstack = f[inputstack[1]]
    elsize, al = get_h5_attributes(fstack)

    outsize = [int(np.ceil(d/b))
               for d,b in zip(fstack.shape, blockreduce)]

    gname = os.path.join(datadir, dset_name + outpf + inputstack[0] + '.h5')
    g = h5py.File(gname, 'w')
    outds = g.create_dataset(inputstack[1], outsize,
                             dtype=fstack.dtype,
                             compression="gzip")

    outds[:,:,:] = block_reduce(fstack,
                                block_size=tuple(blockreduce),
                                func=eval(func))

    elsize = [e*b for e, b in zip(elsize, blockreduce)]
    write_h5_attributes(outds, elsize, al)

    f.close()
    g.close()


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
