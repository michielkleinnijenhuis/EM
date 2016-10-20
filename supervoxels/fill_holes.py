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
    parser.add_argument('-w', '--method', default="1",
                        help='...')
    parser.add_argument('-l', '--labelvolume', default=['_labelMA', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-m', '--labelmask', default=None, nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=None, nargs=2,
                        help='...')
    parser.add_argument('--maskMA', default=None, nargs=2,
                        help='...')
    parser.add_argument('-o', '--outpf_labelvolume', default='_filled',
                        help='...')
    parser.add_argument('-p', '--outpf_holes', default=None,
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    method = args.method
    labelvolume = args.labelvolume
    labelmask = args.labelmask
    maskMM = args.maskMM
    maskMA = args.maskMA
    outpf_labelvolume = args.outpf_labelvolume
    outpf_holes = args.outpf_holes

    labels, elsize = loadh5(datadir, dset_name + labelvolume[0],
                            fieldname=labelvolume[1])
    if labelmask is not None:
        labelmask = loadh5(datadir, dset_name + labelmask[0],
                           fieldname=labelmask[1], dtype='bool')[0]
        labels[~labelmask] = 0
        del(labelmask)

    labels_filled = fill_holes(labels, method)

    writeh5(labels_filled, datadir,
            dset_name + labelvolume[0] + outpf_labelvolume,
            element_size_um=elsize, dtype='int32')

    if maskMA is not None:
#         mask = np.zeros_like(labels_filled, dtype='uint8')
#         mask[labels_filled>0] = 1
        writeh5(labels_filled.astype('bool'), datadir,
                dset_name + maskMA[0] + outpf_labelvolume,
                element_size_um=elsize, dtype='uint8')

    holes = labels_filled
    holes[labels>0] = 0

    if maskMM is not None:
        mask, elsize = loadh5(datadir, dset_name + maskMM[0],
                              fieldname=maskMM[1])
        mask[holes>0] = 0
        writeh5(mask, datadir, dset_name + maskMM[0] + outpf_labelvolume,
                element_size_um=elsize, dtype='uint8')

    if outpf_holes is not None:
        writeh5(holes, datadir, dset_name + labelvolume[0] + outpf_holes,
                element_size_um=elsize, dtype='int32')


# ========================================================================== #
# function defs
# ========================================================================== #


def fill_holes(MA, method):
    """Fill holes in labels."""

    if method == '1':
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

    elif method == "2":

        for l in np.unique(MA)[1:]:
            MA[binary_fill_holes(MA == l)] = l
            MA[binary_closing(MA == l, iterations=10)] = l
            MA[binary_fill_holes(MA == l)] = l

    elif method == "3":

        for l in np.unique(MA)[1:]:
            MA[binary_fill_holes(MA == l)] = l

    return MA


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
