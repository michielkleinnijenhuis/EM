#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.morphology import remove_small_objects, closing, ball


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', default=['_labelMA', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('-o', '--outpf', default='_labelMA_core2D_merged',
                        help='...')
    parser.add_argument('-m', '--maxrange', default=4, type=int,
                        help='...')
    parser.add_argument('-t', '--threshold_overlap', default=0.01, type=float,
                        help='...')
    parser.add_argument('-s', '--min_labelsize', default=10000, type=int,
                        help='...')
    parser.add_argument('-i', '--iterations', default=2, type=int,
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    maskMM = args.maskMM
    outpf = args.outpf
    maxrange = args.maxrange
    threshold_overlap = args.threshold_overlap
    min_labelsize = args.min_labelsize
    iterations = args.iterations

    labels, elsize, al = loadh5(datadir, dset_name + labelvolume[0],
                                fieldname=labelvolume[1])

    for _ in range(0, iterations):
        MAlist = []
        for i in range(0, labels.shape[0] - maxrange):
            for j in range(1, maxrange):
                data_section = labels[i,:,:]
                nb_section = labels[i+j,:,:]
                MAlist = merge_neighbours(MAlist, data_section, nb_section,
                                          threshold_overlap)

        ulabels = np.unique(labels)
        fw = [l if l in ulabels else 0
              for l in range(0, np.amax(labels) + 1)]
        labels = forward_map(np.array(fw), labels, MAlist)

        remove_small_objects(labels, min_size=min_labelsize, in_place=True)

        labels = closing(labels, ball(maxrange))

    if maskMM is not None:
        maskMM = loadh5(datadir, dset_name + maskMM[0],
                        fieldname=maskMM[1], dtype='bool')[0]
        labels[maskMM] = 0

    remove_small_objects(labels, min_size=min_labelsize, in_place=True)

    writeh5(labels, datadir, dset_name + outpf,
            dtype='int32', element_size_um=elsize, axislabels=al)


# ========================================================================== #
# function defs
# ========================================================================== #


def merge_neighbours(MAlist, data_section, nb_section, threshold_overlap=0.01):
    """Adapt the forward map to merge neighbouring labels."""

    data_labels = np.trim_zeros(np.unique(data_section))
    for data_label in data_labels:

        mask_data = data_section == data_label
        bins = np.bincount(nb_section[mask_data])
        if len(bins) <= 1:
            continue

        nb_label = np.argmax(bins[1:]) + 1
        if nb_label == data_label:
            continue
        n_data = np.sum(mask_data)
        n_nb = bins[nb_label]
        if float(n_nb) / float(n_data) < threshold_overlap:
            continue

        MAlist = classify_label(MAlist, set([data_label, nb_label]))

    return MAlist


def classify_label(MAlist, labelset):
    """Add set of labels to an axonset or create new axonset."""

    found = False
    for i, MA in enumerate(MAlist):
        for l in labelset:
            if l in MA:
                MAlist[i] = MA | labelset
                found = True
                break
    if not found:
        MAlist.append(labelset)

    return MAlist


def forward_map(fw, labels, MAlist):
    """Map all labelsets in MAlist to axons."""

    for MA in MAlist:
        MA = sorted(list(MA))
        for l in MA:
            fw[l] = MA[0]

    fwmapped = fw[labels]

    return fwmapped


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
