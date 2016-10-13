#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np


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
    parser.add_argument('-m', '--outpf_mask', default=None,
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    supervoxels = args.supervoxels
    outpf_supervoxels = args.outpf_supervoxels
    outpf_mask = args.outpf_mask

    labels = loadh5(datadir, dset_name + labelvolume[0],
                    fieldname=labelvolume[1])[0]
    ws, elsize = loadh5(datadir, dset_name + supervoxels[0],
                        fieldname=supervoxels[1])

    maskMA = np.zeros_like(ws, dtype='bool')
    wsmaskforlabel = np.zeros_like(ws, dtype='bool')

    ulabels = np.trim_zeros(np.unique(labels))
    for l in ulabels:
        labelmask = labels == l
        svoxs_in_label = np.trim_zeros(np.unique(ws[labelmask]))

        if svoxs_in_label.any():
#             wsmaskforlabel.fill(False)
#             for sv in svoxs_in_label:
#                 wsmaskforlabel[ws == sv] = True
            forward_map = [True if i in svoxs_in_label else False
                           for i in range(0, np.max(ws) + 1)]
            wsmaskforlabel = np.array(forward_map)[ws]

            ws[wsmaskforlabel] = svoxs_in_label[0]
            maskMA[wsmaskforlabel] = True

    writeh5(ws, datadir, dset_name + supervoxels[0] + outpf_supervoxels,
            element_size_um=elsize, dtype='int32')
    writeh5(maskMA, datadir, dset_name + outpf_mask,
            element_size_um=elsize, dtype='uint8')


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
