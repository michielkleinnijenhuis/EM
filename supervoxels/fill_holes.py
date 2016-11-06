#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from scipy.ndimage.morphology import (binary_closing,
                                      binary_fill_holes,
                                      grey_dilation)
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import watershed
from _symtable import DEF_BOUND


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-w', '--method', default="2",
                        help='...')
    parser.add_argument('-L', '--labelvolume', default=['_labelMA', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-m', '--labelmask', nargs=2, default=None,
                        help='...')
    parser.add_argument('--maskDS', nargs=2, default=None,
                        help='...')
    parser.add_argument('--maskMM', nargs=2, default=None,
                        help='...')
    parser.add_argument('--maskMX', nargs=2, default=None,
                        help='...')
    parser.add_argument('--maskMA', nargs=2, default=None,
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_filled', 'stack'],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    method = args.method
    labelvolume = args.labelvolume
    labelmask = args.labelmask
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMX = args.maskMX
    maskMA = args.maskMA
    outpf = args.outpf

    labels, elsize = loadh5(datadir, dset_name + labelvolume[0],
                            fieldname=labelvolume[1])
    if labelmask is not None:
        mask_label = loadh5(datadir, dset_name + labelmask[0],
                           fieldname=labelmask[1], dtype='bool')[0]
        labels[~mask_label] = 0
        del(mask_label)

    masks = None
    if method == "4":
        masks = get_masks(datadir, dset_name, maskDS, maskMM, maskMX)

    labels_filled = fill_holes(method, labels, masks)
    holes = np.copy(labels_filled)
    holes[labels>0] = 0

    writeh5(labels_filled, datadir,
            dset_name + labelvolume[0] + outpf[0],
            element_size_um=elsize, dtype='int32')

    if maskMA is not None:
        writeh5(labels_filled.astype('bool'), datadir,
                dset_name + maskMA[0] + outpf[0],
                element_size_um=elsize, dtype='uint8')

    if maskMM is not None:
        mask, elsize = loadh5(datadir, dset_name + maskMM[0],
                              fieldname=maskMM[1])
        mask[holes>0] = 0
        writeh5(mask, datadir, dset_name + maskMM[0] + outpf[0],
                element_size_um=elsize, dtype='uint8')

    writeh5(holes, datadir, dset_name + labelvolume[0] + outpf[0] + '_holes',
            element_size_um=elsize, dtype='int32')


# ========================================================================== #
# function defs
# ========================================================================== #


def fill_holes(method, labels, masks=None):
    """Fill holes in labels."""

    if method == '1':
        binim = labels != 0
        # does binary_closing bridge seperate labels? YES, and eats from boundary
        # binim = binary_closing(binim, iterations=10)
        holes = label(~binim, connectivity=1)

        labelCount = np.bincount(holes.ravel())
        background = np.argmax(labelCount)
        holes[holes == background] = 0

        labels_dil = grey_dilation(labels, size=(3,3,3))

        rp = regionprops(holes, labels_dil)
        mi = {prop.label: prop.max_intensity for prop in rp}
        fw = [mi[key] if key in mi.keys() else 0
              for key in range(0, np.amax(holes) + 1)]
        fw = np.array(fw)

        holes_remapped = fw[holes]

        labels = np.add(labels, holes_remapped)

    elif method == "2":

        for l in np.unique(labels)[1:]:
            labels[binary_fill_holes(labels == l)] = l
            labels[binary_closing(labels == l, iterations=10)] = l
            labels[binary_fill_holes(labels == l)] = l

    elif method == "3":

        for l in np.unique(labels)[1:]:
            labels[binary_fill_holes(labels == l)] = l

    elif method == "4":

        maskDS, maskMM, maskMX = masks
        MMlabels = fill_holes_watershed(labels, maskDS, maskMM)
        MXlabels = fill_holes_watershed(labels, maskDS, maskMX)
        labels = np.maximum(MMlabels, MXlabels)

    return labels


def fill_holes_watershed(labels, maskDS, maskMM):
    """Fill holes not reachable from unmyelinated axons space."""

    mask = ~maskDS | maskMM | labels.astype('bool')
    labels_mask = label(~mask)

    counts = np.bincount(labels_mask.ravel())
    bg = np.argmax(counts[1:]) + 1

    mask = ~maskDS | maskMM | (labels_mask == bg)
    labels = watershed(mask, labels, mask=~mask)

    return labels


def get_masks(datadir, dset_name, maskDS, maskMM, maskMX):
    """Load the set of masks for method4."""

    maskDS = loadh5(datadir, dset_name + maskDS[0],
                    fieldname=maskDS[1])[0]
    maskMM = loadh5(datadir, dset_name + maskMM[0],
                    fieldname=maskMM[1])[0]
    maskMX = loadh5(datadir, dset_name + maskMX[0],
                    fieldname=maskMX[1])[0]

    return (maskDS, maskMM, maskMX)


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
