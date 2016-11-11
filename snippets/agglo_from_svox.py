#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

from skimage.morphology import remove_small_objects
from skimage.measure import label

def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', default=['_labelMA', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-s', '--supervoxels', default=['_svox', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-o', '--outpf_supervoxels', default='_labelMA',
                        help='...')
    parser.add_argument('-m', '--simple', action='store_true',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    supervoxels = args.supervoxels
    outpf_supervoxels = args.outpf_supervoxels
    simple = args.simple

    labels = loadh5(datadir, dset_name + labelvolume[0],
                    fieldname=labelvolume[1])[0]
    ws, elsize, al = loadh5(datadir, dset_name + supervoxels[0],
                            fieldname=supervoxels[1])

    remove_small_objects(labels, min_size=10000, in_place=True)
    writeh5(labels, datadir, dset_name + supervoxels[0] + outpf_supervoxels + '_delsmall',
            dtype='int32', element_size_um=elsize, axislabels=al)
#     remove_small_objects(ws, min_size=1000, in_place=True)

    if simple:
        pass
#         usvox = np.trim_zeros(np.unique(ws[labels != 0]))
#         forward_map = [True if i in usvox else False
#                        for i in range(0, np.max(ws) + 1)]
#         wsmask = np.array(forward_map)[ws]
#         labels_agglo = label(wsmask)
#         writeh5(labels_agglo, datadir, dset_name + supervoxels[0] + outpf_supervoxels,
#                 dtype='int32', element_size_um=elsize, axislabels=al)
    else:
        usvox = np.trim_zeros(np.unique(ws[labels != 0]))
        MAlist =[]
        for i, svox in enumerate(usvox):
            labels_in_svox = set(np.trim_zeros(np.unique(labels[ws == svox])))
            found = False
            for i, axon in enumerate(MAlist):
                for l in labels_in_svox:
                    if l in axon:
                        MAlist[i] = axon | labels_in_svox
                        found = True
                        print(MAlist[i])
                        break
            if not found:
                MAlist.append(labels_in_svox)

        fw = np.zeros(np.amax(labels) + 1)
        print(len(MAlist))
        for axon in MAlist:
            axon = sorted(list(axon))
            print(axon)
            for l in axon:
                fw[l] = axon[0]

        fwmapped = fw[labels]

        writeh5(fwmapped, datadir, dset_name + supervoxels[0] + outpf_supervoxels,
                dtype='int32', element_size_um=elsize, axislabels=al)


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
