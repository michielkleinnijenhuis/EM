#!/usr/bin/env python

"""Find labels that do not traverse through the volume.

"""

import sys
import argparse
import pickle

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, watershed

from wmem import parse, utils, wmeMPI, LabelImage


def main(argv):
    """Find labels that do not traverse through the volume."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_merge_labels(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    merge_labels(
        args.inputfile,
        args.dataslices,
        args.slicedim,
        args.merge_method,
        args.min_labelsize,
        args.offsets,
        args.overlap_threshold,
        args.searchradius,
        args.data,
        args.maskMM,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def merge_labels(
        image_in,
        dataslices=None,
        slicedim=0,
        merge_method='neighbours',
        min_labelsize=0,
        offsets=2,
        overlap_threshold=0.50,
        searchradius=[100, 30, 30],
        h5path_data='',
        h5path_mmm='',
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Find labels that do not traverse through the volume."""

    mpi = wmeMPI(usempi)

    # Open data for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices)
    comps = im.split_path(outputpath)
    ulabels = np.unique(im.ds[:])
    maxlabel = np.amax(ulabels)
    print(maxlabel)

    labelsets = {}
    labelsets, filled = merge_labels_by_method(merge_method,
                                               mpi,
                                               im,
                                               labelsets,
                                               slicedim,
                                               offsets,
                                               overlap_threshold,
                                               h5path_data,
                                               h5path_mmm,
                                               min_labelsize,
                                               searchradius)

    utils.dump_labelsets(labelsets, comps, mpi.rank)
    if mpi.enabled:
        mpi.comm.Barrier()

    im.close()

    if mpi.rank == 0:

        utils.combine_labelsets(labelsets, comps)

        if filled is not None:  # FIXME: only filled of rank=0 written
            outpaths = {'out': outputpath, 'filled': ''}
            outpaths = utils.gen_steps(outpaths, save_steps)
            mo2 = LabelImage(outpaths['filled'],
                             elsize=im.elsize,
                             axlab=im.axlab,
                             shape=im.dims,
                             chunks=im.chunks,
                             dtype=im.dtype,
                             protective=protective)
            mo2.create()
            fw = [l if l in ulabels else 0 for l in range(0, maxlabel + 1)]
            mo2.write(utils.forward_map(np.array(fw), filled, labelsets))
            mo2.close()

        do_map_labels = True
        if do_map_labels:
            map_labels(im, None, None, outputpath, save_steps, protective)


def map_labels(
        image_in,
        dataslices=None,
        labelsets=None,
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """Map groups of labels to a single label."""

    # Open data for reading.
    im = utils.get_image(image_in, dataslices=dataslices)
    comps = im.split_path(outputpath)

    if labelsets is None:
        lspath = '{}_{}.pickle'.format(comps['base'], comps['dset'])
        labelsets = utils.read_labelsets(lspath)

    # apply forward map
    ulabels = np.unique(im.ds[:])
    maxlabel = np.amax(ulabels)

    mo1 = LabelImage(outputpath,
                     elsize=im.elsize,
                     axlab=im.axlab,
                     shape=im.dims,
                     chunks=im.chunks,
                     dtype=im.dtype,
                     protective=protective)
    mo1.create()
    fw = [l if l in ulabels else 0 for l in range(0, maxlabel + 1)]
    mo1.write(utils.forward_map(np.array(fw), im.ds[:], labelsets))
    if save_steps:
        outpaths = {'out': outputpath, 'stitched': ''}
        outpaths = utils.gen_steps(outpaths, save_steps)
        mo2 = LabelImage(outpaths['stitched'],
                         elsize=im.elsize,
                         axlab=im.axlab,
                         shape=im.dims,
                         chunks=im.chunks,
                         dtype=im.dtype,
                         protective=protective)
        mo2.create()
        fw = np.zeros(maxlabel + 1, dtype='i')
        mo2.write(utils.forward_map(np.array(fw), im.ds[:], labelsets))
        mo2.close()

    mo1.close()
    im.close()


def merge_labels_by_method(merge_method, mpi, labels, labelsets={},
                           slicedim=0, offsets=2,
                           overlap_threshold=0.50,
                           h5path_data='', h5path_mmm='',
                           min_labelsize=10, searchradius=[100, 30, 30]):
    """Find candidate labelsets."""

    # find connection candidates
    if merge_method == 'neighbours':
        labelsets, filled = evaluate_labels(merge_method, mpi,
                                            labels.ds, None,
                                            labelsets, None,
                                            overlap_threshold=overlap_threshold)

    elif merge_method == 'neighbours_slices':
        labelsets, filled = evaluate_labels(merge_method, mpi,
                                            labels.ds, None,
                                            labelsets, None,
                                            overlap_threshold=overlap_threshold,
                                            offsets=offsets,
                                            slicedim=slicedim,
                                            )

    elif merge_method == 'watershed':

        filled = np.copy(labels.ds)  # TODO: copy Image class

        data = utils.get_image(h5path_data, comm=mpi.comm,
                               dataslices=labels.dataslices)
        if h5path_mmm:
            mask = utils.get_image(h5path_mmm, comm=mpi.comm,
                                   dataslices=labels.dataslices)
#                 mask = utils.h5_load(h5path_mmm, load_data=True, dtype='bool')[0]
        else:
            mask = np.zeros_like(data.ds, dtype='bool')

        labelsets, filled = evaluate_labels(merge_method, mpi,
                                            labels.ds, None,
                                            labelsets, filled,
                                            ds_data=data.ds,
                                            ds_mask=mask.ds,
                                            min_labelsize=min_labelsize,
                                            searchradius=searchradius,
                                            )

    elif merge_method == 'conncomp':

        labelmask = labels.ds != 0
        labels_connected = label(labelmask, connectivity=1)
        labelsets, filled = evaluate_labels(merge_method, mpi,
                                            labels_connected, labels.ds,
                                            labelsets, None,
                                            ds_data=data.ds,
                                            ds_mask=mask.ds,
                                            min_labelsize=min_labelsize,
                                            searchradius=searchradius,
                                            )

    return labelsets, filled


def evaluate_labels(merge_method, mpi, labels, aux, labelsets, filled, **kwargs):

    rp = regionprops(labels, aux)

    # Prepare for processing with MPI.
    mpi.nblocks = len(rp)
    mpi.scatter_series()

    for i in mpi.series:
        prop = rp[i]

        if merge_method == 'neighbours':
            labelsets = merge_neighbours(labels, prop, labelsets, filled,
                                         kwargs['overlap_threshold'],
                                         )
        elif merge_method == 'neighbours_slices':
            labelsets = merge_neighbours_slices(labels, prop, labelsets, filled,
                                                kwargs['overlap_threshold'],
                                                kwargs['offsets'],
                                                kwargs['slicedim'],
                                                )
        elif merge_method == 'watershed':
            labelsets, filled = merge_watershed(labels, prop, labelsets, filled,
                                                kwargs['ds_data'],
                                                kwargs['ds_mask'],
                                                kwargs['min_labelsize'],
                                                kwargs['searchradius'],
                                                )
        elif merge_method == 'conncomp':
            labelsets = merge_conncomp(labels, prop, labelsets, filled)

    return labelsets, filled


def merge_neighbours(labels, prop, labelsets={}, filled=None,
                     overlap_thr=20):
    """Find candidates for label merge based on overlapping faces."""

    # get indices to the box surrounding the label
    C = find_region_coordinates('around', labels, prop, [1, 1, 1])
    x, X, y, Y, z, Z = C

    # get a mask of voxels adjacent to the label (boundary)
    imregion = labels[z:Z, y:Y, x:X]
    labelmask = imregion == prop.label
    boundary = binary_dilation(labelmask) - labelmask

    # evaluate which labels overlap sufficiently with this mask
    # TODO: dice-like overlap?
    counts = np.bincount(imregion[boundary])
    label_neighbours = np.argwhere(counts > overlap_thr)  # TODO: dice
    label_neighbours = [l for ln in label_neighbours for l in ln]
    if len(label_neighbours) > 1:
        labelset = set([prop.label] + label_neighbours[1:])
        labelsets = utils.classify_label_set(labelsets, labelset,
                                             prop.label)

    return labelsets


def merge_neighbours_slices(labels, prop, labelsets={}, filled=None,
                            overlap_threshold=0.50, offsets=2, slicedim=0):
    """Find candidates for label merge based on overlapping faces."""

    # get indices to the box surrounding the label
    C = find_region_coordinates('around', labels, prop, [0, 0, 0])
    x, X, y, Y, z, Z = C
#         labels.dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
#         imregion = labels.ds[z:Z, y:Y, x:X]
#         data_section = utils.get_slice(labels.ds, z, slicedim)
#         data_section = utils.get_slice(imregion, 0, slicedim)
    data_section = labels[z, y:Y, x:X]
    data_section[data_section != prop.label] = 0
    for j in range(1, offsets):
        if z-j >= 0:
            nb_section = labels[z-j, y:Y, x:X]
#                 nb_section = utils.get_slice(labels, z-j, slicedim)
            labelsets = merge_neighbours_from_slices(labelsets,
                                                     data_section, nb_section,
                                                     overlap_threshold)

    data_section = labels[Z-1, y:Y, x:X]
    data_section[data_section != prop.label] = 0
    for j in range(1, offsets):
        if Z-1+j < labels.shape[0]:
            nb_section = labels[Z-1+j, y:Y, x:X]
            labelsets = merge_neighbours_from_slices(labelsets,
                                                     data_section, nb_section,
                                                     overlap_threshold)

    return labelsets


def merge_neighbours_from_slices(labelsets, data_section, nb_section,
                                 threshold_overlap=0.50):
    """Adapt the forward map to merge neighbouring labels."""

    data_labels = np.trim_zeros(np.unique(data_section))
    for data_label in data_labels:
        mask_data = data_section == data_label
        bins = np.bincount(nb_section[mask_data])  # find labels with overlap
        if len(bins) <= 1:
            continue

        nb_label = np.argmax(bins[1:]) + 1  # select the largest overlap
        if nb_label == data_label:
            continue

        n_data = np.sum(mask_data)
        mask_nb = nb_section == nb_label
        tp = np.sum(np.logical_and(mask_nb, mask_data))
        dice = tp * 2.0 / (np.sum(mask_nb) + n_data)
#         n_nb = bins[nb_label]
#         if float(n_nb) / float(n_data) < threshold_overlap:
        if dice < threshold_overlap:
            continue

#         print(dice)
        labelsets = utils.classify_label_set(labelsets,
                                             set([data_label, nb_label]))

    return labelsets


def merge_conncomp(labels, prop, labelsets={}, filled=None):
    """Find candidates for label merge based on connected components."""

    counts = np.bincount(prop.intensity_image[prop.image])
    labelset = set(list(np.flatnonzero(counts)))
    if len(counts) > 1:
        labelsets = utils.classify_label_set(labelsets, labelset,
                                             prop.label)

    return labelsets


def merge_watershed(labels, prop, labelsets={}, filled=None,
                    ds_data=None, ds_mask=None,
                    min_labelsize=10, searchradius=[100, 30, 30]):
    """Find candidates for label merge based on watershed."""

    # investigate image region above and below bbox
    for direction in ['down', 'up']:

#         print('processing {}, direction {}'.format(prop.label, direction))

        C = find_region_coordinates(direction, labels, prop, searchradius)
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
            filled[z:Z, y:Y, x:X] = np.copy(wsout)

    return labelsets, filled


def find_candidate_ws(direction, labelsets, prop, imregion,
                      data, mask=None, min_labelsize=10):
    """Find a merge candidate by watershed overlap."""

    wsout = None

    idx = {'down': -1, 'up': 0}[direction]

#     # do the watershed
    if mask is None:
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
                # reset the rest of the region
                mask = ws != prop.label
                wsout[mask] = imregion[mask]

    return labelsets, wsout


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


if __name__ == "__main__":
    main(sys.argv[1:])
