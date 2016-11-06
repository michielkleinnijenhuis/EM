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
    parser.add_argument('-s', '--supervoxels', nargs=2,
                        default=['_svox', 'stack'],
                        help='...')
    parser.add_argument('-l', '--labelset_files', nargs='*', default=[],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMA', 'stack'],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    supervoxels = args.supervoxels
    labelset_files = args.labelset_files
    outpf = args.outpf

    ws, elsize, al = loadh5(datadir, dset_name + supervoxels[0],
                            fieldname=supervoxels[1])

    maxlabel = np.amax(ws)
    print("number of labels in watershed: %d" % maxlabel)
    fw = np.zeros(maxlabel + 1, dtype='i')

    for lsfile in labelset_files:
        labelsets = read_labelsets(lsfile)

    fwmapped = forward_map(np.array(fw), ws, labelsets)

    fstem = dset_name + supervoxels[0] + outpf[0]
    writeh5(fwmapped, datadir, fstem, outpf[1], 'int32', elsize, al)


# ========================================================================== #
# function defs
# ========================================================================== #


def read_labelsets(lsfile):
    """Read labelsets from file."""

    e = os.path.splitext(lsfile)[1]
    if e == '.pickle':
        with open(lsfile) as f:
            labelsets = pickle.load(f)
    else:
        labelsets = read_labelsets_from_txt(lsfile)

    return labelsets


def read_labelsets_from_txt(lsfile):
    """Read labelsets from a textfile."""

    labelsets = {}

    with open(lsfile) as f:
        lines=f.readlines()
        for line in lines:
            splitline = line.split(':', 2)
            lsk = splitline[0]
            lsv = set(np.fromstring(splitline[1], dtype=int, sep=' '))
            labelsets[lsk] = lsv

    return labelsets


def write_labelsets(labelsets, filestem, filetypes='txt'):
    """Write labelsets to file."""

    if 'txt' in filetypes:
        filepath = filestem + '.txt'
        write_labelsets_to_txt(labelsets, filepath)
    if 'pickle' in filetypes:
        filepath = filestem + '.pickle'
        with open(filepath, "wb") as file:
            pickle.dump(labelsets, file)


def write_labelsets_to_txt(labelsets, filepath):
    """Write labelsets to a textfile."""

    with open(filepath, "wb") as file:
        for lsk, lsv in labelsets.items():
            file.write("%8d: " % lsk)
            ls = sorted(list(lsv))
            for l in ls:
                file.write("%8d " % l)
            file.write('\n')


def forward_map(fw, labels, labelsets):
    """Map all labelset in value to key."""

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
