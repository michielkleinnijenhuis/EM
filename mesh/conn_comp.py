#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects, binary_dilation
from skimage.measure import regionprops


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('--maskDS', default=['_maskDS', '/stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', '/stack'], nargs=2,
                        help='...')
    parser.add_argument('-o', '--outpf', default='_labelMA',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    maskDS = args.maskDS
    maskMM = args.maskMM
    outpf = args.outpf

    elsize = loadh5(datadir, dset_name)[1]
    maskDS = loadh5(datadir, dset_name + maskDS[0],
                    fieldname=maskDS[1], dtype='bool')[0]
    maskMM = loadh5(datadir, dset_name + maskMM[0],
                    fieldname=maskMM[1], dtype='bool')[0]

    mask = np.logical_or(binary_dilation(maskMM), ~maskDS)
    remove_small_objects(mask, min_size=100000, in_place=True)

    labels = label(~mask, return_num=False, connectivity=None)
    remove_small_objects(labels, min_size=10000, connectivity=1, in_place=True)

    # remove the unmyelinated axons (largest label)
    rp = regionprops(labels)
    areas = [prop.area for prop in rp]
    labs = [prop.label for prop in rp]
    llab = labs[np.argmax(areas)]
    labels[labels == llab] = 0

    labels = relabel_sequential(labels)[0]

    writeh5(labels, datadir, dset_name + outpf, element_size_um=elsize)


# ========================================================================== #
# function defs
# ========================================================================== #


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
