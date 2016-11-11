#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pickle

import h5py
import numpy as np

from skimage.measure import regionprops


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', nargs=2,
                        default=['_labelMA', 'stack'],
                        help='...')
    parser.add_argument('-s', '--supervoxels', nargs=2,
                        default=['_svox', 'stack'],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMA', 'stack'],
                        help='...')
    parser.add_argument('-t', '--ratio_thr', type=float, default=0,
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    supervoxels = args.supervoxels
    outpf = args.outpf
    ratio_thr = args.ratio_thr

    labels = loadh5(datadir, dset_name + labelvolume[0],
                    fieldname=labelvolume[1])[0]
    ws, elsize, al = loadh5(datadir, dset_name + supervoxels[0],
                            fieldname=supervoxels[1])

    maxlabel = np.amax(ws)
    print("number of labels in watershed: %d" % maxlabel)
    fw = np.zeros(maxlabel + 1, dtype='i')

    areas_ws = np.bincount(ws.ravel())

    labelsets = {}
    rp_lw = regionprops(labels, ws)
    for prop in rp_lw:

        maskedregion = prop.intensity_image[prop.image]
        counts = np.bincount(maskedregion)
        svoxs_in_label = [l for sl in np.argwhere(counts) for l in sl]

        ratios_svox_in_label = [float(counts[svox]) / float(areas_ws[svox])
                                for svox in svoxs_in_label]
        fwmask = np.greater(ratios_svox_in_label, ratio_thr)
        labelset = np.array(svoxs_in_label)[fwmask]
        labelsets[prop.label] = set(labelset) - set([0])

    fstem = dset_name + outpf[0] + supervoxels[0]
    filestem = os.path.join(datadir, fstem + "_svoxsets")
    write_labelsets(labelsets, filestem, filetypes=['txt', 'pickle'])

    fwmapped = forward_map(np.array(fw), ws, labelsets)
    writeh5(fwmapped, datadir, fstem, outpf[1], 'int32', elsize, al)


# ========================================================================== #
# function defs
# ========================================================================== #


def write_labelsets(labelsets, filestem, filetypes='txt'):
    """Write labelsets to file."""

    if 'txt' in filetypes:
        filepath = filestem + '.txt'
        write_labelsets_to_txt(labelsets, filepath)
    if 'pickle' in filetypes:
        filepath = filestem + '.pickle'
        with open(filepath, 'wb') as file:
            pickle.dump(labelsets, file)


def write_labelsets_to_txt(labelsets, filepath):
    """Write labelsets to a textfile."""

    with open(filepath, 'wb') as file:
        for lsk, lsv in labelsets.items():
            file.write("%8d: " % lsk)
            ls = sorted(list(lsv))
            for l in ls:
                file.write("%8d " % l)
            file.write('\n')


def forward_map(fw, labels, labelsets):
    """Map all labelsets to axons."""

    for lsk, lsv in labelsets.items():
        lsv = sorted(list(lsv))
        for l in lsv:
            fw[l] = lsk

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
