#!/usr/bin/env python

"""Delete/split/merge/... labels in a labelvolume.

"""

import sys
import argparse
import os

import numpy as np
from skimage.measure import label, regionprops

from wmem import parse, utils


def main(argv):
    """Delete/split/merge/... labels in a labelvolume."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_remap_labels(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    remap_labels(
        args.inputfile,
        args.delete_labels,
        args.delete_files,
        args.except_files,
        args.merge_labels,
        args.merge_files,
        args.split_labels,
        args.split_files,
        args.aux_labelvolume,
        args.min_labelsize,
        args.min_segmentsize,
        args.keep_only_largest,
        args.conncomp,
        args.nifti_output,
        args.nifti_transpose,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def remap_labels(inputfile,
                 delete_labels=[], delete_files=[], except_files=[],
                 merge_labels=[], merge_files=[],
                 split_labels=[], split_files=[],
                 aux_labelvolume='',
                 min_labelsize=0, min_segmentsize=0,
                 keep_only_largest=False, conncomp=False,
                 nifti_output=False, nifti_transpose=False,
                 h5path_out='', save_steps=False, protective=False):
    """Delete/split/merge/... labels in a labelvolume."""

    # check output paths
    outpaths = {'out': h5path_out, 'deleted': '', 'split': '', 'merged': ''}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.load(inputfile)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(ds_in[:], ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # filter labels on size
    root = os.path.splitext(h5file_out.filename)[0]
    ls_small = utils.filter_on_size(ds_out[:], min_labelsize, False,
                                    save_steps, root, ds_out.name[1:],
                                    outpaths, elsize, axlab)[1]
    delete_labels = set(delete_labels) | ls_small

    # delete labels
    ds_out[:] = delete_labelsets(ds_out[:], delete_labels, delete_files,
                                 except_files)
    if save_steps:  # FIXME!
        save_diff(ds_in[:], ds_out[:], 'deleted', outpaths, elsize, axlab)

    # split labels
    ds_out[:] = split_labelsets(ds_out[:], split_labels, split_files,
                                aux_labelvolume, conncomp)
    if save_steps:
        save_diff(ds_in[:], ds_out[:], 'split', outpaths, elsize, axlab)

    # merge labels
    ds_out[:] = merge_labelsets(ds_out[:], merge_labels, merge_files)
    if save_steps:
        save_diff(ds_in[:], ds_out[:], 'merged', outpaths, elsize, axlab)

    # remove small, non-contiguous segments of labels
    if min_segmentsize or keep_only_largest:
        ds_out[:] = filter_segments(ds_out[:], min_segmentsize,
                                    keep_only_largest)

    if nifti_output:
        if nifti_transpose:
            ds_out[:] = np.transpose(ds_out[:])
            elsize = elsize[::-1]
            axlab = axlab[::-1]
        fpath = '{}_{}.nii.gz'.format(root, ds_out.name[1:])
        utils.write_to_nifti(fpath, ds_out[:], elsize)

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def delete_labelsets(labels, delete_labels=[], delete_files=[],
                     except_files=[]):
    """Delete labels from a labelvolume."""

    # add labels in delete_files to delete_labels list
    for delfile in delete_files:
        delsets = utils.read_labelsets(delfile)
        dellist = [d for _, dv in delsets.items() for d in list(dv)]
        delete_labels = set(dellist) | set(delete_labels)
    if delete_labels:
        # remove labels that are in except files
        for excfile in except_files:
            excsets = utils.read_labelsets(excfile)
            exclist = [d for _, dv in excsets.items() for d in list(dv)]
            delete_labels = delete_labels - set(exclist)
        # delete the labels
        print('deleting ', delete_labels)
        ulabels = np.unique(labels)
        fw = [l if ((l in ulabels) and
                    (l not in delete_labels)) else 0
              for l in range(0, np.amax(ulabels) + 1)]
        labels = np.array(fw)[labels]

    return labels, fw


def split_labelsets(labels, split_labels=[], split_files=[],
                    aux_labelvolume='', conncomp=False):
    """Split labels in a labelvolume."""

    for splfile in split_files:
        splsets = utils.read_labelsets(splfile)
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
                imregion = labels[z:Z, y:Y, x:X]
                if aux_labelvolume:
                    uimregion = np.unique(prop.intensity_image[prop.image])
                    nullmask = prop.image
                    for uim in uimregion:
                        nullmask = nullmask | (imregion == uim)
                    nullmask = nullmask - prop.image
                    imregion[nullmask] = 0
                imregion[mask] = split_label[mask] + maxlabel
                maxlabel += num

    return labels


def merge_labelsets(labels, merge_labels=[], merge_files=[]):
    """Merge labels in a labelvolume."""

    if merge_files:
        ulabels = np.unique(labels)
        fw = [l if l in ulabels else 0
              for l in range(0, np.amax(ulabels) + 1)]
        for merfile in merge_files:
            mersets = utils.read_labelsets(merfile)
            print('merging ', mersets)
            labels = utils.forward_map(np.array(fw), labels, mersets)

    if merge_labels:
        print('merging ', merge_labels)
        merge_labels = np.reshape(np.array(merge_labels), (-1, 2))
        ulabels = np.unique(labels)
        fw = [l if ((l in ulabels) and
                    (l not in merge_labels) and
                    (np.remainder(np.argwhere(merge_labels == l), 2) == 0))
              else merge_labels[np.argwhere(merge_labels == l) + 1]
              for l in range(0, np.amax(ulabels) + 1)]
        labels = np.array(fw)[labels]

    return labels, fw


def filter_segments(labels, min_segmentsize=0, keep_only_largest=False):
    """Remove small, non-contiguous segments of labels."""

    rp = regionprops(labels)
    for prop in rp:
        z, y, x, Z, Y, X = tuple(prop.bbox)
        mask = prop.image
        split_label = label(mask, return_num=True)[0]
        counts = np.bincount(split_label.ravel())

        if keep_only_largest:
            if len(counts) > 2:
                largest = np.argmax(counts[1:]) + 1
                imregion = labels[z:Z, y:Y, x:X]
                nullmask = np.zeros_like(prop.image)
                for i in range(1, len(counts)):
                    if (i != largest):
                        nullmask = nullmask | (split_label == i)
        elif min_segmentsize:
            nulllabels = [l for sl in np.argwhere(counts < min_segmentsize)
                          for l in sl]
            if nulllabels:
                imregion = labels[z:Z, y:Y, x:X]
                nullmask = np.zeros_like(prop.image)
                for nl in nulllabels:
                    nullmask = nullmask | (split_label == nl)

            imregion[nullmask] = 0

    return labels


def save_diff(orig_labels, labels, stepname, outpaths, elsize, axlab):
    """Save the difference between original and remapped labelvolume."""

    diff = np.zeros_like(orig_labels)
    diffmask = orig_labels != labels
    diff[diffmask] = labels[diffmask]
    utils.save_step(outpaths, stepname, diff, elsize, axlab)


if __name__ == "__main__":
    main(sys.argv[1:])
