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


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-w', '--methods', default="2",
                        help='...')
    parser.add_argument('-q', '--min_labelsize', type=int, default=10,
                        help='...')
    parser.add_argument('-s', '--selem', nargs='*', type=int, default=[3,3,3],
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
    methods = args.methods
    selem = args.selem
    labelvolume = args.labelvolume
    labelmask = args.labelmask
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMX = args.maskMX
    maskMA = args.maskMA
    outpf = args.outpf

    labels, elsize, al = loadh5(datadir, dset_name + labelvolume[0],
                                labelvolume[1])

    if labelmask is not None:
        mask_label = loadh5(datadir, dset_name + labelmask[0],
                           labelmask[1], 'bool')[0]
        labels[~mask_label] = 0
        del(mask_label)

    maskinfo = datadir, dset_name, maskDS, maskMM, maskMX

    labels_filled = np.copy(labels)
    for m in methods:
        labels_filled = fill_holes(m, labels_filled, selem, maskinfo)

    holes = np.copy(labels_filled)
    holes[labels>0] = 0

    writeh5(labels_filled, datadir,
            dset_name + labelvolume[0] + outpf[0], outpf[1],
            'int32', elsize, al)
    writeh5(holes, datadir,
            dset_name + labelvolume[0] + outpf[0] + '_holes', outpf[1],
            'int32', elsize, al)

    if maskMA is not None:
        writeh5(labels_filled.astype('bool'), datadir,
                dset_name + maskMA[0] + outpf[0], outpf[1],
                'uint8', elsize, al)

    if maskMM is not None:
        mask = loadh5(datadir, dset_name + maskMM[0],
                      fieldname=maskMM[1])[0]
        mask[holes>0] = 0
        writeh5(mask, datadir, dset_name + maskMM[0] + outpf[0], outpf[1],
                'uint8', elsize, al)


# ========================================================================== #
# function defs
# ========================================================================== #


def fill_holes(method, labels, selem=[3, 3, 3], maskinfo=None):
    """Fill holes in labels."""

    if method == '1':
        binim = labels != 0
        # does binary_closing bridge seperate labels? YES, and eats from boundary
        # binim = binary_closing(binim, iterations=10)
        holes = label(~binim, connectivity=1)

        labelCount = np.bincount(holes.ravel())
        background = np.argmax(labelCount)
        holes[holes == background] = 0

        labels_dil = grey_dilation(labels, size=selem)

        rp = regionprops(holes, labels_dil)
        mi = {prop.label: prop.max_intensity for prop in rp}
        fw = [mi[key] if key in mi.keys() else 0
              for key in range(0, np.amax(holes) + 1)]
        fw = np.array(fw)

        holes_remapped = fw[holes]

        labels = np.add(labels, holes_remapped)

    elif method == "2":

        rp = regionprops(labels)
        for prop in rp:
            print(prop.label)
            z, y, x, Z, Y, X = tuple(prop.bbox)
            mask = prop.image
#             mask = binary_fill_holes(mask)
            mask = binary_closing(mask, iterations=selem[0])
            mask = binary_fill_holes(mask)
            imregion = labels[z:Z,y:Y,x:X]
            imregion[mask] = prop.label

    elif method == "3":

        rp = regionprops(labels)
        for prop in rp:
            print(prop.label)
            z, y, x, Z, Y, X = tuple(prop.bbox)
            mask = prop.image
            mask = binary_fill_holes(mask)
            imregion = labels[z:Z,y:Y,x:X]
            imregion[mask] = prop.label

    elif method == "4":

        datadir, dset_name, maskDS, maskMM, maskMX = maskinfo

        mask = get_mask(datadir, dset_name, maskDS, maskMM)
        MMlabels = fill_holes_watershed(labels, mask)

        mask = get_mask(datadir, dset_name, maskDS, maskMX)
        MXlabels = fill_holes_watershed(labels, mask)

        labels = np.maximum(MMlabels, MXlabels)

    return labels


def fill_holes_watershed(labels, mask_in):
    """Fill holes not reachable from unmyelinated axons space."""

    mask = mask_in | labels.astype('bool')
    labels_mask = label(~mask)

    counts = np.bincount(labels_mask.ravel())
    bg = np.argmax(counts[1:]) + 1

    mask = mask_in | (labels_mask == bg)
    labels = watershed(mask, labels, mask=~mask)

    return labels


def get_mask(datadir, dset_name, maskDS, maskMM):
    """Load the set of masks for method4."""

    maskDS = loadh5(datadir, dset_name + maskDS[0], maskDS[1])[0]
    maskMM = loadh5(datadir, dset_name + maskMM[0], maskMM[1])[0]

    mask = ~maskDS | maskMM

    return mask


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
    main(sys.argv[1:])
