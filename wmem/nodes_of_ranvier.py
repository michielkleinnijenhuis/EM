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
        args.merge_methods,
        args.overlap_threshold,
        args.data,
        args.maskMM,
        args.searchradius,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def nodes_of_ranvier(
        h5path_in,
        min_labelsize=0,
        remove_small_labels=False,
        h5path_boundarymask='',
        merge_methods=['neighbours'],
        overlap_threshold=20,
        h5path_data='',
        h5path_mmm='',
        searchradius=[100, 30, 30],
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Find labels that do not traverse through the volume."""

    # check output paths
    outpaths = {'out': h5path_out,
                'largelabels': '', 'smalllabelmask': '',
                'boundarymask': '',
                'labels_nt': '', 'labels_tv': '',
                'filled': '',
                }
    root, ds_main = outpaths['out'].split('.h5')
    for dsname, outpath in outpaths.items():
        grpname = ds_main + "_steps"
        outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)
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

    # start with the set of all labels
    ulabels = np.unique(labels)
    maxlabel = np.amax(ulabels)
    labelset = set(ulabels)
    print("number of labels in labelvolume: {}".format(len(labelset)))

    # get the labelsets that touch the borders
#     sidesmask = get_boundarymask(h5path_boundarymask, ('ero', 3))
#     sidesmask = get_boundarymask(h5path_boundarymask)
    top_margin = 1  # 4 or 14
    bot_margin = 1  # 4
    sidesmask = get_boundarymask(h5path_boundarymask, ('invdil', 3), top_margin, bot_margin)
    ls_bot = set(np.unique(labels[:bot_margin, :, :]))
    ls_top = set(np.unique(labels[-top_margin:, :, :]))
    ls_sides = set(np.unique(labels[sidesmask]))
    ls_border = ls_bot | ls_top | ls_sides
    ls_centre = labelset - ls_border
    # get the labels that do not touch the border twice
    ls_bts = (ls_bot ^ ls_top) ^ ls_sides
    ls_tbs = (ls_top ^ ls_bot) ^ ls_sides
    ls_sbt = (ls_sides ^ ls_bot) ^ ls_top
    ls_nt = ls_centre | ls_bts | ls_tbs | ls_sbt

    # filter labels on size
    root = os.path.splitext(h5file_out.filename)[0]
    ls_small = utils.filter_on_size(labels, labelset, min_labelsize,
                                    remove_small_labels,
                                    save_steps, root, ds_out.name[1:],
                                    outpaths, elsize, axlab)[2]
    labelset -= ls_small
    ls_nt -= ls_small
    ls_short = filter_on_heigth(labels, 0)  # 5
    labelset -= ls_short
    ls_nt -= ls_short
    ls_tv = labelset - ls_nt

    print('number of large, long labels: {}'.format(len(labelset)))
    print('number of large, long in-volume labels: {}'.format(len(ls_nt)))
    print('number of large, long through-volume labels: {}'.format(len(ls_tv)))

    labelsets = {l: set([l]) for l in ls_tv}
    filestem = '{}_{}_tv_auto'.format(root, ds_out.name[1:])
    utils.write_labelsets(labelsets, filestem, filetypes=['txt'])
    labelsets = {l: set([l]) for l in ls_nt}
    filestem = '{}_{}_nt_auto'.format(root, ds_out.name[1:])
    utils.write_labelsets(labelsets, filestem, filetypes=['txt'])

    # map the large labels that don't traverse the volume
    fw_nt = np.zeros(maxlabel + 1, dtype='i')
    for l in ls_nt:
        fw_nt[l] = l
    labels_nt = fw_nt[labels]

    # automated label merge
    labelsets = {}
#     min_labelsize = 10
    if 0:
        labelsets, filled = merge_labels(labels_nt, labelsets, merge_methods,
                                         overlap_threshold,
                                         h5path_data, h5path_mmm,
                                         min_labelsize,
                                         searchradius)
#         fw = np.zeros(maxlabel + 1, dtype='i')
        ds_out[:] = utils.forward_map(np.array(fw_nt), labels, labelsets)
    else:
        filled = None

#     fw = np.zeros(maxlabel + 1, dtype='i')
    ds_out[:] = utils.forward_map(np.array(fw_nt), labels, labelsets)

    if save_steps:

        utils.save_step(outpaths, 'boundarymask', sidesmask, elsize, axlab)

        utils.save_step(outpaths, 'labels_nt', labels_nt, elsize, axlab)

        fw_tv = np.zeros(maxlabel + 1, dtype='i')
        for l in ls_tv:
            fw_tv[l] = l
        labels_tv = fw_tv[labels]
        utils.save_step(outpaths, 'labels_tv', labels_tv, elsize, axlab)

        if filled is not None:
            fw = np.zeros(maxlabel + 1, dtype='i')
            filled = utils.forward_map(np.array(fw), filled, labelsets)
            utils.save_step(outpaths, 'filled', filled, elsize, axlab)

        filestem = '{}_{}_automerged'.format(root, ds_out.name[1:])
        utils.write_labelsets(labelsets, filestem,
                              filetypes=['txt', 'pickle'])

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def get_boundarymask(h5path_mask, masktype=('invdil', 7),
                     top_margin=4, bot_margin=4):
    """Load or generate a mask."""

    mask = utils.h5_load(h5path_mask, load_data=True, dtype='bool')[0]
    if masktype[0] == 'ero':
        mask = binary_erosion(mask, ball(masktype[1]))
    elif masktype[0] == 'invdil':
        mask = scipy_binary_dilation(~mask, iterations=masktype[1], border_value=0)
        mask[:bot_margin, :, :] = False
        mask[-top_margin:, :, :] = False

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


def merge_labels(labels, labelsets={}, merge_methods=[],
                 overlap_threshold=20,
                 h5path_data='', h5path_mmm='',
                 min_labelsize=10, searchradius=[100, 30, 30]):
    """Find candidate labelsets."""

    # find connection candidates
    for merge_method in merge_methods:

        if merge_method == 'neighbours':
            labelsets = merge_neighbours(labels, labelsets,
                                         overlap_threshold)
            filled = None

        if merge_method == 'neighbours_slices':
            labelsets = merge_neighbours_slices(labels, labelsets,
                                                overlap_threshold)
            filled = None

        elif merge_method == 'conncomp':
            labelsets = merge_conncomp(labels, labelsets)
            filled = None

        elif merge_method == 'watershed':
            labelsets, filled = merge_watershed(labels, labelsets,
                                                h5path_data, h5path_mmm,
                                                min_labelsize, searchradius)

    return labelsets, filled


def merge_neighbours(labels, labelsets={}, overlap_thr=20):
    """Find candidates for label merge based on overlapping faces."""

    rp_nt = regionprops(labels)

    for prop in rp_nt:

        # get indices to the box surrounding the label
        C = find_region_coordinates('around', labels, prop, [1, 1, 1])
        x, X, y, Y, z, Z = C

        # get a mask of voxels adjacent to the label (boundary)
        imregion = labels[z:Z, y:Y, x:X]
        labelmask = imregion == prop.label
        boundary = np.logical_xor(binary_dilation(labelmask), labelmask)

        # evaluate which labels overlap sufficiently with this mask
        # TODO: dice-like overlap?
        counts = np.bincount(imregion[boundary])
        label_neighbours = np.argwhere(counts > overlap_thr)
        label_neighbours = [l for ln in label_neighbours for l in ln]
        if len(label_neighbours) > 1:
            labelset = set([prop.label] + label_neighbours[1:])
            labelsets = utils.classify_label_set(labelsets, labelset,
                                                 prop.label)

    return labelsets


def merge_neighbours_slices(labels, labelsets={}, overlap_thr=20):
    """Find candidates for label merge based on overlapping faces."""

    from wmem.merge_slicelabels import merge_neighbours
    overlap_thr = 0.20
    offsets = 2

    rp_nt = regionprops(labels)

    for prop in rp_nt:
        # get indices to the box surrounding the label
        C = find_region_coordinates('around', labels, prop, [0, 0, 0])
        x, X, y, Y, z, Z = C

        data_section = labels[z, y:Y, x:X]
        data_section[data_section != prop.label] = 0
        for j in range(1, offsets):
            if z-j >= 0:
                nb_section = labels[z-j, y:Y, x:X]
                labelsets = merge_neighbours(labelsets,
                                             data_section, nb_section,
                                             threshold_overlap=overlap_thr)

        data_section = labels[Z-1, y:Y, x:X]
        data_section[data_section != prop.label] = 0
        for j in range(1, offsets):
            if Z-1+j < labels.shape[0]:
                nb_section = labels[Z-1+j, y:Y, x:X]
                labelsets = merge_neighbours(labelsets,
                                             data_section, nb_section,
                                             threshold_overlap=overlap_thr)

    return labelsets


def merge_conncomp(labels, labelsets={}):
    """Find candidates for label merge based on connected components."""

    # binarize labelvolume and relabel for connected components
    labelmask = labels != 0
    labels_connected = label(labelmask, connectivity=1)

    # find the original labels contained in each connected component
    # TODO: detection of non-contiguous components in the original?
    rp = regionprops(labels_connected, labels)
    for prop in rp:
        counts = np.bincount(prop.intensity_image[prop.image])
        labelset = set(list(np.flatnonzero(counts)))
        if len(counts) > 1:
            labelsets = utils.classify_label_set(labelsets, labelset,
                                                 prop.label)

    return labelsets


def merge_watershed(labels, labelsets={},
                    h5path_data='', h5path_mmm='',
                    min_labelsize=10, searchradius=[100, 30, 30]):
    """Find candidates for label merge based on watershed."""

    rp_nt = regionprops(labels)
    labels_filled = np.copy(labels)

    ds_data = utils.h5_load(h5path_data, load_data=True)[0]
    if h5path_mmm:
        ds_mask = utils.h5_load(h5path_mmm, load_data=True, dtype='bool')[0]
    else:
        ds_mask = np.zeros_like(ds_data, dtype='bool')

    for prop in rp_nt:
        # investigate image region above and below bbox
        for direction in ['down', 'up']:
            print('processing {}, direction {}'.format(prop.label, direction))

            C = find_region_coordinates(direction, labels,
                                        prop, searchradius)
            x, X, y, Y, z, Z = C
            if ((z == 0) or (z == labels.shape[0] - 1)):
                continue

            # TODO: improve searchregion by projecting along axon direction
            # TODO: improve searchregion by not taking the groundplane of the whole label region
            imregion = labels[z:Z, y:Y, x:X]
            labels_in_region = np.unique(imregion)
#             print(labels_in_region)

            if len(labels_in_region) < 2:
                continue  # label 0 and prop.label assumed to be there

            labelsets, wsout = find_candidate_ws(direction, labelsets, prop,
                                                 imregion,
                                                 ds_data[z:Z, y:Y, x:X],
                                                 ds_mask[z:Z, y:Y, x:X],
                                                 min_labelsize)

            if wsout is not None:
                labels_filled[z:Z, y:Y, x:X] = np.copy(wsout)

    return labelsets, labels_filled


def find_candidate_ws(direction, labelsets, prop, imregion,
                      data, maskMM, min_labelsize=10):
    """Find a merge candidate by watershed overlap."""

    wsout = None

    idx = {'down': -1, 'up': 0}[direction]

#     # do the watershed
    mask = np.ones_like(imregion, dtype='bool')
    mask[data < 0.25] = False
    seeds = np.zeros_like(imregion, dtype='int')
    seeds[idx, :, :][imregion[idx, :, :] == prop.label] = prop.label
    seeds[idx, :, :][imregion[idx, :, :] != prop.label] = -1
    ws = watershed(-data, seeds, mask=mask)

#     """NOTE:
#     seeds are in the borderslice,
#     with the current label as prop.label
#     (watershedded to fill the full axon),
#     the maskMM as background, and the surround as negative label
#     """
#     # TODO: don't use -data; make it more general
#     # TODO: implement string_mask?
#     # fill the seedslice (outside of the myelin compartment)
#     seeds = np.zeros_like(imregion, dtype='int')
#     seeds[idx, :, :] = watershed(-data[idx, :, :],
#                                  imregion[idx, :, :],
#                                  mask=~maskMM[idx, :, :])
#     # set all non-prop.label voxels to -1
#     seeds[idx, :, :][seeds[idx, :, :] != prop.label] = -1
#     # set the myelin voxels to 0
#     seeds[idx, :, :][maskMM[idx, :, :]] = 0
#     # do the watershed
#     ws = watershed(-data, seeds, mask=~maskMM)

    rp_ws = regionprops(ws, imregion)  # no 0 in rp
    labels_ws = [prop_ws.label for prop_ws in rp_ws]
    try:
        # select the watershed-object of the current label
        idx = labels_ws.index(prop.label)
    except ValueError:
        pass
    else:
        # get the overlap (voxel count) of labels within the watershed object
        counts = np.bincount(imregion[rp_ws[idx].image])
        if len(counts) > 1:
            # select the largest candidate overlapping the watershed
            # TODO: improve criteria for accepting candidate
            candidate = np.argmax(counts[1:]) + 1
            # only select it if the overlap is larger than min_labelsize
            if ((counts[candidate] > min_labelsize) and
                    (candidate != prop.label)):
                print('merging {} and {}'.format(prop.label, candidate))
                labelset = set([prop.label, candidate])
                labelsets = utils.classify_label_set(labelsets, labelset,
                                                     prop.label)
                wsout = ws
                mask = ws != prop.label
                wsout[mask] = imregion[mask]

    return labelsets, wsout


def filter_on_heigth(labels, min_height, ls_short=set([])):

    rp_nt = regionprops(labels)
    for prop in rp_nt:
        if prop.bbox[3]-prop.bbox[0] <= min_height:
            ls_short |= set([prop.label])
    print('number of short labels: {}'.format(len(ls_short)))

    return ls_short


def correct_NoR(image_in):
    """Add a manually-defined set of labels to through-volume and remove from not-through."""

    from wmem import LabelImage

    # read the labelvolume
    im = utils.get_image(image_in, imtype='Label')
    comps = im.split_path()

    # map and write the nt and tv volumes
    def write_vol(outputpath, im, ls):
        mo = LabelImage(outputpath, **im.get_props())
        mo.create()
        mo.write(im.forward_map(labelsets=ls, from_empty=True))
        mo.close()

    # pop manual tv-labels from auto-nt; add to auto-tv; write to tv/nt;
    ls_stem = '{}_{}_NoR'.format(comps['base'], comps['dset'])
    nt = utils.read_labelsets('{}_{}.txt'.format(ls_stem, 'nt_auto'))
    tv = utils.read_labelsets('{}_{}.txt'.format(ls_stem, 'tv_auto'))
    tv_man = utils.read_labelsets('{}_{}.txt'.format(ls_stem, 'tv_manual'))
    for l in tv_man[0]:
        nt.pop(l)
        tv[l] = set([l])

    for ls_name, ls in zip(['nt', 'tv'], [nt, tv]):
        utils.write_labelsets(ls, '{}_{}'.format(ls_stem, ls_name), filetypes=['txt'])
        dset_out = '{}_NoR_steps/labels_{}'.format(comps['dset'], ls_name)
        outputpath = os.path.join(comps['file'], dset_out)
        write_vol(outputpath, im, ls)

    im.close()


if __name__ == "__main__":
    main(sys.argv[1:])
