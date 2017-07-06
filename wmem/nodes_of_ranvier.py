#!/usr/bin/env python

"""Find labels that do not traverse through the volume.

"""

import sys
import argparse
import os

from scipy.ndimage.morphology import binary_dilation as scipy_binary_dilation
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, ball, watershed

from wmem import parse, utils


# TODO: write elsize and axislabels
def main(argv):
    """Find labels that do not traverse through the volume."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_nodes_of_ranvier(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    nodes_of_ranvier(
        args.inputfile,
        args.min_labelsize,
        args.remove_small_labels,
        args.boundarymask,
        args.merge_method,
        args.overlap_threshold,
        args.data,
        args.maskMM,
        args.searchradius,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def nodes_of_ranvier(h5path_in,
                     min_labelsize=0, remove_small_labels=False,
                     h5path_boundarymask='',
                     merge_methods=['neighbours'],
                     overlap_threshold=20,
                     h5path_data='', h5path_mmm='', searchradius=[100, 30, 30],
                     h5path_out='', save_steps=False, protective=False):
    """Find labels that do not traverse through the volume."""

    # check output paths
    outpaths = {'out': h5path_out,
                'filled': '', 'largelabels': '',
                'boundarymask': '', 'labels_NoR': ''}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
    labels = ds_in[:]  # FIXME: do we make a copy, or use ds_out?

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # filter labels on size
    root = os.path.splitext(h5file_out.filename)[0]
    labels = utils.filter_on_size(labels, min_labelsize,
                                  remove_small_labels,
                                  save_steps, root, ds_out.name[1:],
                                  outpaths, elsize, axlab)[0]

    # start with the set of all labels (larger than min_labelsize)
    ulabels = np.unique(labels)
    maxlabel = np.amax(ulabels)
    print("number of labels in labelvolume: %d" % maxlabel)
    labelset = set(ulabels)

    # get a mask of the sides of the dataset
    sidesmask = get_boundarymask(h5path_boundarymask, save_steps)
    if save_steps and (not h5path_boundarymask):
        utils.save_step(outpaths, 'boundarymask', labels, elsize, axlab)

    # get the labelsets that touch the borders
    ls_bot = set(np.unique(labels[0, :, :]))
    ls_top = set(np.unique(labels[-1, :, :]))
    ls_sides = set(np.unique(labels[sidesmask]))
    ls_border = ls_bot | ls_top | ls_sides
    ls_centre = labelset - ls_border
    # get the labels that do not touch the border twice
    ls_bts = (ls_bot ^ ls_top) ^ ls_sides
    ls_tbs = (ls_top ^ ls_bot) ^ ls_sides
    ls_sbt = (ls_sides ^ ls_bot) ^ ls_top
    # get the labels that do not pass through the volume
    ls_nt = ls_centre | ls_bts | ls_tbs | ls_sbt

    # map the labels that do not traverse the volume
    fw = np.zeros(maxlabel + 1, dtype='i')
    for l in ls_nt:
        fw[l] = l
    labels_nt = fw[labels]

    # see if we need to do merging of labels
    if set(merge_methods) & set(['neighbours', 'conncomp', 'watershed']):
        if save_steps:
            utils.save_step(outpaths, 'labels_NoR', labels_nt, elsize, axlab)
    else:
        ds_out[:] = labels_nt
        return

    # find connection candidates
    for merge_method in merge_methods:
        if merge_method == 'neighbours':
            labelsets = merge_neighbours(labels, labels_nt, overlap_threshold)

        elif merge_method == 'conncomp':
            labelsets = merge_conncomp(labels_nt)

        elif merge_method == 'watershed':
            labelsets, filled = merge_watershed(labels, labels_nt,
                                                h5path_data, h5path_mmm,
                                                min_labelsize, searchradius)
            if save_steps:
                data = utils.forward_map(np.array(fw), filled, labelsets)
                utils.save_step(outpaths, 'filled', data, elsize, axlab)

        # TODO merge labelsets of iterations

    root = os.path.splitext(h5file_out.filename)[0]
    filestem = '{}_{}_automerged'.format(root, ds_out.name[1:])
    utils.write_labelsets(labelsets, filestem, filetypes=['txt', 'pickle'])

    ds_out[:] = utils.forward_map(np.array(fw), labels, labelsets)

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def get_boundarymask(h5path_mask, masktype='invdil'):
    """Load or generate a mask."""

    if h5path_mask:
        mask = utils.h5_load(h5path_mask, load_data=True, dtype='bool')[0]
    else:
        if masktype == 'ero':
            mask = binary_erosion(mask, ball(3))
        elif masktype == 'invdil':
            mask = scipy_binary_dilation(~mask, iterations=1, border_value=0)
            mask[:4, :, :] = 0
            mask[-4:, :, :] = 0

    return mask


def find_region_coordinates(direction, labels, prop, searchradius):
    """Find coordinates of a box bordering a partial label."""

    """NOTE:
    prop.bbox is in half-open interval
    """
    if direction == 'around':  # a wider box around the label's bbox
        z = max(0, int(prop.bbox[0]) - searchradius[0])
        Z = min(labels.shape[0], int(prop.bbox[3]) + searchradius[0])
        y = max(0, int(prop.bbox[1]) - searchradius[1])
        Y = min(labels.shape[1], int(prop.bbox[4]) + searchradius[1])
        x = max(0, int(prop.bbox[2]) - searchradius[2])
        X = min(labels.shape[2], int(prop.bbox[5]) + searchradius[2])

        return (x, X, y, Y, z, Z)

    # get the z-range of a box above/below the label's bbox
    elif direction == 'down':  # a box below the label bbox
        borderslice = int(prop.bbox[0])
        z = max(0, borderslice - searchradius[0])
        Z = borderslice
    elif direction == 'up':  # a box above the label bbox
        borderslice = int(prop.bbox[3]) - 1
        z = borderslice
        Z = min(labels.shape[0], borderslice + searchradius[0])
    # find the centroid of the label within the borderslice
    labels_slc = np.copy(labels[borderslice, :, :])
    labels_slc[labels_slc != prop.label] = 0
    rp_bs = regionprops(labels_slc)
    ctrd = rp_bs[0].centroid
    # get the x,y-range of a box above/below the label's bbox
    y = max(0, int(ctrd[0]) - searchradius[1])
    Y = min(labels.shape[1], int(ctrd[0]) + searchradius[1])
    x = max(0, int(ctrd[1]) - searchradius[2])
    X = min(labels.shape[2], int(ctrd[1]) + searchradius[2])

    return (x, X, y, Y, z, Z)


def merge_neighbours(labels, labels_nt, overlap_thr=20):
    """Find candidates for label merge based on overlapping faces."""

    labelsets = {}

    rp_nt = regionprops(labels_nt)

    for prop in rp_nt:

        # get indices to the box surrounding the label
        C = find_region_coordinates('around', labels, prop, [1, 1, 1])
        x, X, y, Y, z, Z = C

        # get a mask of voxels adjacent to the label (boundary)
        imregion = labels_nt[z:Z, y:Y, x:X]
        labelmask = imregion == prop.label
        boundary = binary_dilation(labelmask) - labelmask

        # evaluate which labels overlap sufficiently with this mask
        counts = np.bincount(imregion[boundary])
        label_neighbours = np.argwhere(counts > overlap_thr)
        label_neighbours = [l for ln in label_neighbours for l in ln]
        if len(label_neighbours) > 1:
            labelset = set([prop.label] + label_neighbours[1:])
            labelsets = utils.classify_label_set(labelsets, labelset,
                                                 prop.label)

    return labelsets


def merge_conncomp(labels_nt):
    """Find candidates for label merge based on connected components."""

    labelsets = {}

    # binarize labelvolume and relabel for connected components
    labelmask = labels_nt != 0
    labels_connected = label(labelmask, connectivity=1)

    # find the original labels contained in each connected component
    # TODO: detection of non-contiguous components in the original?
    rp = regionprops(labels_connected, labels_nt)
    for prop in rp:
        counts = np.bincount(prop.intensity_image[prop.image])
        labelset = set(list(np.flatnonzero(counts)))
        if len(counts) > 1:
            labelsets = utils.classify_label_set(labelsets, labelset,
                                                 prop.label)

    return labelsets


def merge_watershed(labels, labels_nt,
                    h5path_data, h5path_mmm,
                    min_labelsize=0, searchradius=[100, 30, 30]):
    """Find candidates for label merge based on watershed."""

    labelsets = {}

    rp_nt = regionprops(labels_nt)
    labels_filled = np.copy(labels_nt)

    ds_data = utils.h5_load(h5path_data, load_data=True)[0]
    ds_mask = utils.h5_load(h5path_mmm, load_data=True, dtype='bool')[0]

    for prop in rp_nt:
        # investigate image region above and below bbox
        for direction in ['down', 'up']:

            C = find_region_coordinates(direction, labels,
                                        prop, searchradius)
            x, X, y, Y, z, Z = C
            if ((z == 0) or (z == labels.shape[0] - 1)):
                continue

            imregion = labels_nt[z:Z, y:Y, x:X]
            labels_in_region = np.unique(imregion)
            # label 0 and prop.label assumed to be always there
            if len(labels_in_region) < 2:
                continue

            labelsets, wsout = find_candidate_ws(direction, labelsets, prop,
                                                 imregion,
                                                 ds_data[z:Z, y:Y, x:X],
                                                 ds_mask[z:Z, y:Y, x:X],
                                                 min_labelsize)

            if wsout is not None:
                labels_filled[z:Z, y:Y, x:X] = np.copy(wsout)

    return labelsets, labels_filled


def find_candidate_ws(direction, labelsets, prop, imregion,
                      data, maskMM, min_labelsize):
    """Find a merge candidate by watershed overlap."""

    wsout = None

    """NOTE:
    seeds are in the borderslice,
    with the current label as prop.label (watershedded to fill the full axon),
    the maskMM as background, and the surround as negative label
    """
    if direction == 'down':
        idx = -1
    elif direction == 'up':
        idx = 0
    seeds = np.zeros_like(imregion)
    # TODO: don't use -data; make it more general
    # TODO: implement string_mask?
    # fill the seedslice (outside of the myelin compartment)
    seeds[idx, :, :] = watershed(-data[idx, :, :],
                                 imregion[idx, :, :],
                                 mask=~maskMM[idx, :, :])
    # set all non-prop.label voxels to -1
    seeds[idx, :, :][seeds[idx, :, :] != prop.label] = -1
    # set the myelin voxels to 0
    seeds[idx, :, :][maskMM[idx, :, :]] = 0

    # do the watershed
    ws = watershed(-data, seeds, mask=~maskMM)

    rp_ws = regionprops(ws, imregion)  # no 0 in rp
    labels_ws = [prop_ws.label for prop_ws in rp_ws]
    try:
        idx = labels_ws.index(prop.label)
    except ValueError:
        pass
    else:
        counts = np.bincount(imregion[rp_ws[idx].image])
        if len(counts) > 1:
            # select the largest candidate overlapping the watershed
            candidate = np.argmax(counts[1:]) + 1
            # only select it if it a proper region
            if counts[candidate] > min_labelsize:
                labelset = set([prop.label, candidate])
                labelsets = utils.classify_label_set(labelsets, labelset,
                                                     prop.label)
                wsout = ws
                mask = ws != prop.label
                wsout[mask] = imregion[mask]

    return labelsets, wsout


if __name__ == "__main__":
    main(sys.argv[1:])
