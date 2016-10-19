#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.measure import label
from scipy.ndimage.morphology import binary_closing, binary_fill_holes, grey_dilation
from skimage.measure import regionprops


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', default=['_labelMA', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-o', '--outpf_labelvolume', default='_filled',
                        help='...')
    parser.add_argument('-w', '--method', default="1",
                        help='...')
    parser.add_argument('-m', '--outpf_holes', default=None,
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    outpf_labelvolume = args.outpf_labelvolume
    outpf_holes = args.outpf_holes
    method = args.method

    labels, elsize = loadh5(datadir, dset_name + labelvolume[0],
                            fieldname=labelvolume[1])

    if method == "1":
        labels_filled = fill_holes_method1(labels)
    elif method == "2":
        labels_filled = fill_holes_method2(labels)
    elif method == "3":
        labels_filled = fill_holes_method3(labels)

    writeh5(labels_filled, datadir, dset_name + labelvolume[0] + outpf_labelvolume,
            element_size_um=elsize, dtype='int32')

    holes = labels_filled
    holes[labels>0] = 0
    writeh5(holes, datadir, dset_name + labelvolume[0] + outpf_holes,
            element_size_um=elsize, dtype='int32')

    # TODO: updated myelin mask


# ========================================================================== #
# function defs
# ========================================================================== #


def fill_holes_method1(MA):
    """Fill holes in labels."""

    binim = MA != 0
    # does this bridge seperate MA's? YES, and eats from boundary
    # binim = binary_closing(binim, iterations=10)
    holes = label(~binim, connectivity=1)

    labelCount = np.bincount(holes.ravel())
    background = np.argmax(labelCount)
    holes[holes == background] = 0

    labels_dil = grey_dilation(MA, size=(3,3,3))

    rp = regionprops(holes, labels_dil)
    mi = {prop.label: prop.max_intensity for prop in rp}
    fw = [mi[key] if key in mi.keys() else 0
          for key in range(0, np.amax(holes) + 1)]
    fw = np.array(fw)

    holes_remapped = fw[holes]

    MA = np.add(MA, holes_remapped)

    return MA


def fill_holes_method2(MA):
    """Fill holes in labels."""

    ulabels = np.unique(MA)[1:]
    for l in ulabels:
        # fill holes
        labels = label(MA!=l)[0]
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        MA[labels != background] = l
        # closing
        binim = MA==l
        binim = binary_closing(binim, iterations=10)
        MA[binim] = l
        # fill holes
        labels = label(MA!=l)[0]
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        MA[labels != background] = l

    return MA


def fill_holes_method3(MA):
    """Fill holes in labels."""

    ulabels = np.unique(MA)[1:]
    labels_filled = np.copy(MA)
    for l in ulabels:
        filled = binary_fill_holes(MA == l)
        labels_filled[filled] = l

    return labels_filled


def loadh5(datadir, dname, fieldname='stack', dtype=None):
    """"""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')
    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]
    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None):
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
    g.close()


if __name__ == "__main__":
    main(sys.argv[1:])
