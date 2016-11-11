#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pickle

import h5py
import numpy as np
import nibabel as nib

from skimage.measure import label, regionprops
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
                        default=[],
                        help='...')
    parser.add_argument('-d', '--delete_files', nargs='*', default=[],
                        help='...')
    parser.add_argument('-e', '--except_files', nargs='*', default=[],
                        help='...')
    parser.add_argument('-M', '--merge_labels', nargs='*', type=int,
                        default=[],
                        help='...')
    parser.add_argument('-m', '--merge_files', nargs='*', default=[],
                        help='...')
    parser.add_argument('-S', '--split_labels', nargs='*', type=int,
                        default=[],
                        help='...')
    parser.add_argument('-s', '--split_files', nargs='*', default=[],
                        help='...')
    parser.add_argument('-A', '--aux_labelvolume', default=[],
                        nargs=2,
                        help='...')
    parser.add_argument('-q', '--min_labelsize', type=int, default=None,
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_proofread', 'stack'],
                        help='...')
    parser.add_argument('-p', '--min_segmentsize', type=int, default=None,
                        help='...')
    parser.add_argument('-k', '--keep_only_largest', action='store_true',
                        help='...')
    parser.add_argument('-n', '--nifti_input', action='store_true',
                        help='...')
    parser.add_argument('-N', '--nifti_output', action='store_true',
                        help='...')
    parser.add_argument('-O', '--output_diff', action='store_true',
                        help='...')
    parser.add_argument('-C', '--conncomp', action='store_true',
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
    delete_files = args.delete_files
    except_files = args.except_files
    merge_labels = args.merge_labels
    merge_files = args.merge_files
    split_labels = args.split_labels
    split_files = args.split_files
    aux_labelvolume = args.aux_labelvolume
    min_labelsize = args.min_labelsize
    min_segmentsize = args.min_segmentsize
    outpf = args.outpf
    keep_only_largest = args.keep_only_largest
    nifti_input = args.nifti_input
    nifti_output = args.nifti_output
    inlayout = args.inlayout
    outlayout = args.outlayout
    output_diff = args.output_diff
    conncomp = args.conncomp

    labels, elsize, al = read_vol(datadir, dset_name, labelvolume, inlayout)

    orig_labels = np.array(labels)

    if delete_files:
        for delfile in delete_files:
            delsets = read_labelsets(delfile)
            dellist = [d for _, dv in delsets.items() for d in list(dv)]
            delete_labels = set(dellist) | set(delete_labels)
    if delete_labels:
        if min_labelsize is not None:
            ls_small = filter_on_size(datadir, dset_name, outpf, 
                                      labels, min_labelsize)[1]
            delete_labels = delete_labels | ls_small
        if except_files:
            for excfile in except_files:
                excsets = read_labelsets(excfile)
                exclist = [d for _, dv in excsets.items() for d in list(dv)]
                delete_labels = delete_labels - set(exclist)

        print('deleting ', delete_labels)

        ulabels = np.unique(labels)

        fw = [l if ((l in ulabels) and
                    (l not in delete_labels)) else 0
              for l in range(0, np.amax(ulabels) + 1)]
        labels = np.array(fw)[labels]

        if output_diff:  # FIXME!
            diff = np.zeros_like(orig_labels)
            diffmask = orig_labels != labels
            diff[diffmask] = labels[diffmask]
            fstem = dset_name + labelvolume[0] + outpf[0] + '_deleted'
            writeh5(diff, datadir, fstem, outpf[1],
                    dtype='int32', element_size_um=elsize, axislabels=al)

    if split_files:
        for splfile in split_files:
            splsets = read_labelsets(splfile)
            spllist = [d for _, sv in splsets.items() for d in list(sv)]
            split_labels = spllist + split_labels
    if split_labels:
        print('splitting ', split_labels)

        ulabels = np.unique(labels)
        maxlabel = np.amax(ulabels)

        if aux_labelvolume:
            aux_labels = read_vol(datadir, dset_name,
                                  aux_labelvolume, inlayout)[0]
        if not conncomp:
            # get the labels from different labelvolume
            for sl in split_labels:
                mask = labels == sl
                aux_ulabels = np.trim_zeros(np.unique(aux_labels[mask]))
                print(sl, aux_ulabels)
                for au in aux_ulabels:
                    if au == sl:
                        continue
                    aux_mask = aux_labels == au
                    if au in ulabels:
                        maxlabel += 1
                        au = maxlabel
                    labels[aux_mask] = au
        else:
            # this relabeling method requires that labels have been disconnected manually
            if aux_labelvolume:
                rp = regionprops(aux_labels, labels)
                print(len(rp))
            else:
                rp = regionprops(labels)
                rp = [prop for prop in rp if prop.label in split_labels]
            for prop in rp:
                print(prop.label)
                z, y, x, Z, Y, X = tuple(prop.bbox)
                mask = prop.image
                split_label, num = label(mask, return_num=True)
                imregion = labels[z:Z,y:Y,x:X]
                if aux_labelvolume:
                    uimregion = np.unique(prop.intensity_image[prop.image])
                    nullmask = prop.image
                    for uim in uimregion:
                        nullmask = nullmask | (imregion == uim)
                    nullmask = nullmask - prop.image
                    imregion[nullmask] = 0
                imregion[mask] = split_label[mask] + maxlabel
                maxlabel += num

        if output_diff:
            diff = np.zeros_like(orig_labels)
            diffmask = orig_labels != labels
            diff[diffmask] = labels[diffmask]
            fstem = dset_name + labelvolume[0] + outpf[0] + '_split'
            writeh5(diff, datadir, fstem, outpf[1],
                    dtype='int32', element_size_um=elsize, axislabels=al)

    if merge_files:
        ulabels = np.unique(labels)
        fw = [l if l in ulabels else 0
              for l in range(0, np.amax(ulabels) + 1)]
        for merfile in merge_files:
            mersets = read_labelsets(merfile)
            print('merging ', mersets)
            labels = forward_map(np.array(fw), labels, mersets)

        if output_diff:
            diff = np.zeros_like(orig_labels)
            diffmask = orig_labels != labels
            diff[diffmask] = labels[diffmask]
            fstem = dset_name + labelvolume[0] + outpf[0] + '_merged'
            writeh5(diff, datadir, fstem, outpf[1],
                    dtype='int32', element_size_um=elsize, axislabels=al)


    if merge_labels:
        print('merging ', merge_labels)
        merge_labels = np.reshape(np.array(merge_labels), (-1, 2))
        ulabels = np.unique(labels)
        fw = [l if ((l in ulabels) and 
                    (l not in merge_labels) and
                    (np.remainder(np.argwhere(merge_labels==l), 2) == 0))
              else merge_labels[np.argwhere(merge_labels==l) + 1]
              for l in range(0, np.amax(ulabels) + 1)]
        labels = np.array(fw)[labels]

    if min_segmentsize is not None:
        rp = regionprops(labels)
        for prop in rp:
            z, y, x, Z, Y, X = tuple(prop.bbox)
            mask = prop.image
            split_label, num = label(mask, return_num=True)
            counts = np.bincount(split_label.ravel())
            nulllabels = [l for sl in np.argwhere(counts < min_segmentsize)
                          for l in sl]
            if nulllabels:
                imregion = labels[z:Z,y:Y,x:X]
                uimregion = np.unique(imregion[prop.image])
                nullmask = np.zeros_like(prop.image)
                for nl in nulllabels:
                    nullmask = nullmask | (split_label == nl)
                imregion[nullmask] = 0

    if keep_only_largest:
        rp = regionprops(labels)
        for prop in rp:
            z, y, x, Z, Y, X = tuple(prop.bbox)
            mask = prop.image
            split_label, num = label(mask, return_num=True)
            counts = np.bincount(split_label.ravel())
            if len(counts) > 2:
                largest = np.argmax(counts[1:]) + 1
                imregion = labels[z:Z,y:Y,x:X]
                uimregion = np.unique(imregion[prop.image])
                nullmask = np.zeros_like(prop.image)
                for i in range(1, len(counts)):
                    if (i != largest):
                        nullmask = nullmask | (split_label == i)
                imregion[nullmask] = 0

    if nifti_output:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        if elsize is not None:
            mat[0][0] = elsize[0]
            mat[1][1] = elsize[1]
            mat[2][2] = elsize[2]
        nib.Nifti1Image(labels, mat).to_filename(dset_name + labelvolume[0] + outpf[0])

    # TODO: handle general case
    if inlayout != outlayout:
        labels = np.transpose(labels)
        elsize = elsize[::-1]
        al = al[::-1]

    fstem = dset_name + labelvolume[0] + outpf[0]
    writeh5(labels, datadir, fstem, outpf[1], 'int32', elsize, al)


# ========================================================================== #
# function defs
# ========================================================================== #


def read_vol(datadir, dset_name, vol, inlayout='xyz'):

    if vol[0].endswith('.nii') or vol[0].endswith('.nii.gz'):
        fname = dset_name + vol[0]
        fpath = os.path.join(datadir, fname)
        img = nib.load(fpath)
        labels = img.get_data().transpose()
        elsize = list(img.header.get_zooms())
        al = inlayout
    else:
        labels, elsize, al = loadh5(datadir, dset_name + vol[0],
                                    fieldname=vol[1])

    return labels, elsize, al


def read_labelsets(lsfile):
    """Read labelsets from file."""

    e = os.path.splitext(lsfile)[1]
    if e == '.pickle':
        with open(lsfile, 'rb') as f:
            labelsets = pickle.load(f)
    else:
        labelsets = read_labelsets_from_txt(lsfile)

    return labelsets


def read_labelsets_from_txt(lsfile):
    """Read labelsets from a textfile."""

    labelsets = {}

    with open(lsfile, 'r') as f:
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
        with open(filepath, 'wb') as file:
            pickle.dump(labelsets, file)


def write_labelsets_to_txt(labelsets, filepath):
    """Write labelsets to a textfile."""

    with open(filepath, 'wb') as file:
        for lsk, lsv in labelsets.items():
            file.write("%8d: " % lsk)
            ls = sorted(list(lsv))
            for l in ls:
                file.write("%10d" % l)
            file.write('\n')


def filter_on_size(datadir, dset_name, outpf, labels, min_labelsize,
                   apply_to_labels=False, write_to_file=True):
    """Filter small labels from a volume; write the set to file."""

    areas = np.bincount(labels.ravel())
    fwmask = areas < min_labelsize

    ls_small = set([l for sl in np.argwhere(fwmask) for l in sl])

    if write_to_file:
        labelsets = {0: ls_small}
        filename = dset_name + outpf[0] + "_smalllabels"
        filestem = os.path.join(datadir, filename)
        write_labelsets(labelsets, filestem, filetypes=['txt', 'pickle'])

    if apply_to_labels:
        smalllabelmask = np.array(fwmask, dtype='bool')[labels]
        labels[smalllabelmask] = 0

    return labels, ls_small


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
