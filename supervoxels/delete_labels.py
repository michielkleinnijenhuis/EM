#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
import nibabel as nib

from skimage.measure import label
from skimage.morphology import remove_small_objects


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-L', '--labelvolume', default=['_labelMA', 'stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-D', '--delete_labels', nargs='*', type=int,
                        default=None,
                        help='...')
    parser.add_argument('-M', '--merge_labels', nargs='*', type=int,
                        default=None,
                        help='...')
    parser.add_argument('-S', '--split_labels', nargs='*', type=int,
                        default=None,
                        help='...')
    parser.add_argument('-s', '--min_labelsize', type=int, default=None,
                        help='...')
    parser.add_argument('-o', '--outpf', default='_proofread',
                        help='...')
    parser.add_argument('-n', '--nifti_input', action='store_true',
                        help='...')
    parser.add_argument('-N', '--nifti_output', action='store_true',
                        help='...')
    parser.add_argument('-i', '--inlayout',
                        help='the data layout of the input')
    parser.add_argument('-l', '--outlayout',
                        help='the data layout for output')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    delete_labels = args.delete_labels
    merge_labels = args.merge_labels
    split_labels = args.split_labels
    min_labelsize = args.min_labelsize
    outpf = args.outpf
    nifti_input = args.nifti_input
    nifti_output = args.nifti_output
    inlayout = args.inlayout
    outlayout = args.outlayout

    if nifti_input:
        fname = dset_name + labelvolume[0] + '.nii.gz'
        fpath = os.path.join(datadir, fname)
        img = nib.load(fpath)
        labels = img.get_data()
        elsize = list(img.header.get_zooms())
        al = inlayout
    else:
        labels, elsize, al = loadh5(datadir, dset_name + labelvolume[0],
                                    fieldname=labelvolume[1])

    # TODO: use forward mapping
    if delete_labels is not None:
        ulabels = np.unique(labels)
        fw = [l if ((l in ulabels) and 
                    (l not in delete_labels)) else 0
              for l in range(0, np.amax(ulabels) + 1)]
        labels = np.array(fw)[labels]

    if merge_labels is not None:
        merge_labels = np.reshape(np.array(merge_labels), (-1, 2))
        ulabels = np.unique(labels)
        fw = [l if ((l in ulabels) and 
                    (l not in merge_labels) and
                    (np.remainder(np.argwhere(merge_labels==l), 2) == 0))
              else merge_labels[np.argwhere(merge_labels==l) + 1]
              for l in range(0, np.amax(ulabels) + 1)]
        labels = np.array(fw)[labels]

    if split_labels is not None:
        for sl in split_labels:
            # this requires that labels have been disconnected manually
            mask = labels == sl
            maxlabel = np.amax(labels)
            split_label = label(mask)
            if min_labelsize is not None:
                remove_small_objects(split_label,
                                     min_size=min_labelsize,
                                     in_place=True)
    #         labels = np.add(labels, split_label + maxlabel)
            for u in np.unique(split_label):
                maxlabel += 1
                split_label[split_label == u] = maxlabel
            labels[mask] = split_label[mask]

    if nifti_output:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        if elsize is not None:
            mat[0][0] = elsize[0]
            mat[1][1] = elsize[1]
            mat[2][2] = elsize[2]
        nib.Nifti1Image(labels, mat).to_filename(dset_name + labelvolume[0] + outpf + '.nii.gz')

    # TODO: handle general case
    if inlayout != outlayout:
        labels = np.transpose(labels)
        elsize = elsize[::-1]
        al = al[::-1]

    writeh5(labels, datadir, dset_name + labelvolume[0] + outpf,
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
