#!/usr/bin/env python

"""Find labels that do not traverse through the volume.

"""

import sys
import argparse
import pickle

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, watershed, disk

from wmem import parse, utils, wmeMPI, LabelImage, MaskImage


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
        args.maskDS,
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
        h5path_mds='',
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Find labels that do not traverse through the volume."""

    mpi = wmeMPI(usempi)

    im = utils.get_image(image_in, imtype='Label',
                         comm=mpi.comm, dataslices=dataslices)

    labelsets = merge_labels_by_method(
        merge_method, mpi, im, slicedim, offsets,
        overlap_threshold,
        h5path_data, h5path_mmm,  h5path_mds,
        min_labelsize, searchradius, outputpath,
        )

    comps = im.split_path(outputpath)
    utils.dump_labelsets(labelsets, comps, mpi.rank)
    if mpi.enabled:
        mpi.comm.Barrier()

    im.close()

    if mpi.rank == 0:

        labelsets = utils.combine_labelsets(labelsets, comps)

        do_map_labels = False
        if do_map_labels:
            im.slices = None
            im.set_slices()
            map_labels(im, None, labelsets, outputpath, save_steps, protective)


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
    im = utils.get_image(image_in, imtype='Label', dataslices=dataslices)
    props = im.get_props(protective=protective)
    comps = im.split_path(outputpath)

    if labelsets is None:
        lspath = '{}_{}.pickle'.format(comps['base'], comps['dset'])
        print(lspath)
        labelsets = utils.read_labelsets(lspath)

    mo1 = LabelImage(outputpath, **props)
    mo1.create()
#     print(labelsets)
    mo1.write(im.forward_map(labelsets=labelsets, from_empty=False))

    if save_steps:
        outpaths = {'out': outputpath, 'stitched': ''}
        outpaths = utils.gen_steps(outpaths, save_steps)
        mo2 = LabelImage(outpaths['stitched'], **props)
        mo2.create()
        mo2.write(im.forward_map(labelsets=labelsets, from_empty=True))
        mo2.close()

    mo1.close()
    im.close()


def merge_labels_by_method(merge_method, mpi, labels,
                           slicedim=0, offsets=2,
                           overlap_threshold=0.50,
                           h5path_data='', h5path_mmm='', h5path_mds='',
                           min_labelsize=10, searchradius=[100, 30, 30],
                           outputpath=''):
    """Find candidate labelsets."""

    # find connection candidates
    if merge_method == 'neighbours':
        labelsets = evaluate_labels(merge_method, mpi, labels, None,
                                    overlap_threshold=overlap_threshold,
                                    )

    elif merge_method == 'neighbours_slices':
        labelsets = evaluate_labels(merge_method, mpi, labels, None,
                                    overlap_threshold=overlap_threshold,
                                    offsets=offsets,
                                    slicedim=slicedim,
                                    )

    elif merge_method == 'watershed':

        data = utils.get_image(h5path_data,
                               comm=mpi.comm, slices=labels.slices)
        if h5path_mmm:
            maskMM = utils.get_image(h5path_mmm, imtype='Mask',
                                     comm=mpi.comm, slices=labels.slices)
        else:
            maskMM = None
        if h5path_mds:
            maskDS = utils.get_image(h5path_mds, imtype='Mask',
                                     comm=mpi.comm, slices=labels.slices)
        else:
            maskDS = None

        labelsets = evaluate_labels(merge_method, mpi, labels, None,
                                    data=data,
                                    maskMM=maskMM,
                                    maskDS=maskDS,
                                    min_labelsize=min_labelsize,
                                    searchradius=searchradius,
                                    outputpath=outputpath,
                                    )

    elif merge_method == 'conncomp':

        props = labels.get_props()
        conn = LabelImage(path='', **props)
        conn.create(comm=mpi.comm)

        labelmask = labels.ds != 0
        conn.ds[:] = label(labelmask, connectivity=1)
        labelsets = evaluate_labels(merge_method, mpi, conn, labels.ds)

    return labelsets


def evaluate_labels(merge_method, mpi, labels, aux, **kwargs):

    labelsets = {}

    rp = regionprops(labels.ds, aux)
    rp_map = {region.label: region for region in rp}

    # Prepare for processing with MPI.
    mpi.nblocks = len(rp)
    mpi.scatter_series()  # randomize=True

    if 'outputpath' in kwargs.keys():
        mo = LabelImage(kwargs['outputpath'], **labels.get_props())
        mo.create()
        mo.write(labels.ds[:])

    for i in mpi.series:
        prop = rp[i]
#     for _, prop in rp_map.items():

        # if prop.label not in [7160]:
        #     continue
        print(prop.label)

        if merge_method == 'neighbours':
            labelsets = merge_neighbours(labels, prop, labelsets,
                                         kwargs['overlap_threshold'],
                                         )
        elif merge_method == 'neighbours_slices':
            labelsets = merge_neighbours_slices(labels, prop, labelsets,
                                                kwargs['overlap_threshold'],
                                                kwargs['offsets'],
                                                kwargs['slicedim'],
                                                )
        elif merge_method == 'watershed':
            labelsets = merge_watershed(mo, rp_map, prop, labelsets,
                                        kwargs['data'],
                                        kwargs['maskMM'],
                                        kwargs['maskDS'],
                                        kwargs['min_labelsize'],
                                        kwargs['searchradius'],
                                        )
        elif merge_method == 'conncomp':
            labelsets = merge_conncomp(labels, prop, labelsets)

    mo.close()

    return labelsets


def merge_neighbours(labels, prop, labelsets={}, overlap_thr=20):
    """Find candidates for label merge based on overlapping faces."""

    # get indices to the box surrounding the label
    slices, _ = get_region_slices_around(labels, prop, [1, 1, 1])

    # get a mask of voxels adjacent to the label (boundary)
    labels.slices = slices
    imregion = labels.slice_dataset(squeeze=False)
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


def merge_neighbours_slices(labels, prop, labelsets={},
                            overlap_threshold=0.50, offsets=2, slicedim=0):
    """Find candidates for label merge based on overlapping faces."""

    # get indices to the box surrounding the label
    _, C = get_region_slices_around(labels, prop, [0, 0, 0])
    x, X, y, Y, z, Z = C

    data_section = labels.ds[z, y:Y, x:X]
    data_section[data_section != prop.label] = 0
    for j in range(1, offsets):
        if z-j >= 0:
            nb_section = labels.ds[z-j, y:Y, x:X]
            labelsets = merge_neighbours_from_slices(labelsets,
                                                     data_section, nb_section,
                                                     overlap_threshold)

    data_section = labels.ds[Z-1, y:Y, x:X]
    data_section[data_section != prop.label] = 0
    for j in range(1, offsets):
        if Z-1+j < labels.dims[0]:
            nb_section = labels.ds[Z-1+j, y:Y, x:X]
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
        # n_nb = bins[nb_label]
        # if float(n_nb) / float(n_data) < threshold_overlap:
        if dice < threshold_overlap:
            continue

        # print(dice)
        labelsets = utils.classify_label_set(labelsets,
                                             set([data_label, nb_label]))

    return labelsets


def merge_conncomp(labels, prop, labelsets={}):
    """Find candidates for label merge based on connected components."""

    counts = np.bincount(prop.intensity_image[prop.image])
    labelset = set(list(np.flatnonzero(counts)))
    if len(counts) > 1:
        labelsets = utils.classify_label_set(labelsets, labelset,
                                             prop.label)

    return labelsets


def merge_watershed(labels, rp_map, prop, labelsets={},
                    data=None, maskMM=None, maskWS=None,
                    min_labelsize=10, searchradius=[100, 30, 30]):
    """Find candidates for label merge based on watershed."""

    # investigate image region above and below bbox
    for direction in ['down', 'up']:

        slices = get_region_slices_projected(direction, labels, prop,
                                             searchradius, z_extent=10)[0]

        # print(prop.label, direction, slices)

        if ((slices[0].start == 0) or (slices[0].start == labels.dims[0] - 1)):
            # print('skipping', prop.label, direction, slices)
            continue

        labels.slices = data.slices = slices
        labels_ds = labels.slice_dataset(squeeze=False)
        data_ds = data.slice_dataset(squeeze=False)
        if maskMM is not None:
            maskMM.slices = slices
            maskMM_ds = maskMM.slice_dataset(squeeze=False)
            maskMM_ds = ~maskMM_ds.astype('bool')
        else:
            maskMM_ds = np.ones_like(labels_ds[:], dtype='bool')
        if maskWS is not None:
            maskWS.slices = slices
            maskWS_ds = maskWS.slice_dataset(squeeze=False)
            maskWS_ds = maskWS_ds.astype('bool')
        else:
            maskWS_ds = np.ones_like(labels_ds[:], dtype='bool')

        if len(np.unique(labels_ds)) < 2:
            continue  # label 0 and prop.label assumed to be there

        # TODO: erode maskMM? doesn't help much;
        # actually loses quite some fills, but haven't dilated the borderslice label
#         mask = binary_dilation(~maskMM_ds.astype('bool'))
        # TODO: add tv to maskMM?? and other labels?
        # TODO: remove mito from maskMM

        if maskWS is not None:
            # TODO: might mark as separate labelset (with key -1)
            # imregion_mask = imregion > 0
            pass

        seeds = create_seeds(labels_ds, prop.label, prop.label, cylinder=0, labval=prop.label)
        ws = watershed(-data_ds, seeds, mask=maskWS_ds)
        labelsets, picked = find_candidate_ws(direction, labelsets, rp_map, prop,
                                                labels_ds, ws, min_labelsize)

        if picked is not None:
            # update labels
            labels.ds[labels.ds[:]==picked] = prop.label
            # update rp_map and prop!
#             label_imgN = np.copy(labels.ds)
#             label_imgN[label_imgN != prop.label] = 0

#             rp_N = regionprops(labels.ds)
#             rp_map_N = {region.label: region for region in rp_N}
#             rp_map[prop.label] = rp_map_N[prop.label]
            rp_map.pop(picked)

            # fill between labels prop.label and picked
            connect_split_label(prop, labels, labels, data, maskMM_ds, searchradius, axons=None)
            # make sure new label is contiguous
            check_split_label(prop, labels, checkonly=False)

    return labelsets


def find_candidate_ws(direction, labelsets, rp_map, prop, imregion, ws,
                      min_labelsize=10):
    """Find a merge candidate by watershed overlap."""

    rp_ws = regionprops(ws, imregion)  # no 0 or -1 in rp
    labels_ws = [prop_ws.label for prop_ws in rp_ws]
    try:
        # select the watershed-object of the current label
        idx_lab = labels_ws.index(prop.label)
    except ValueError:
        pass
    else:
        # get the overlap (voxel count) of labels within the watershed object
        prop_ws = rp_ws[idx_lab]
        counts = np.bincount(prop_ws.intensity_image[prop_ws.image])
        # mark anything other than zero's in the region
        candidate_labels = list(np.argwhere(counts[1:]) + 1)
        # print(candidate_labels)

        if not candidate_labels:
            return labelsets, None

        # reject candidates
        for c in list(candidate_labels):
            c = c[0]
            candidate = rp_map[c]
            # reject self
            if c == prop.label:
                candidate_labels.remove(c)
                counts[c] = 0
            # reject candidates with overlap smaller than min_labelsize
            elif counts[c] < min_labelsize:
                candidate_labels.remove(c)
                counts[c] = 0
            # reject candidates with overlap in z-range
            # FIXME: it's still possible to have overlap if connecting more than 2 labels as the label is not updated after connection
            # possibly pop a label from a list of candidate connections when it has been connected for a certain direction???
            elif direction == 'up' and prop.bbox[3] > candidate.bbox[0]:
                candidate_labels.remove(c)
                counts[c] = 0
            elif direction == 'down' and prop.bbox[0] < candidate.bbox[3]:
                candidate_labels.remove(c)
                counts[c] = 0

        # if there are any candidates left
        if len(candidate_labels) > 0:
            # select the largest candidate overlapping the watershed
            picked = np.argmax(counts[1:]) + 1
#             print('merging {} and {}'.format(prop.label, picked))
            labelset = set([prop.label, picked])
            labelsets = utils.classify_label_set(labelsets, labelset, prop.label)

            # TODO: improve criteria for accepting candidate
            # TODO: pop label+direction from list if merged?
            # TODO: blocky fills when not enough space around the label for sufficient negative label
            # TODO: include boundary as possible candidate

            return labelsets, picked

        else:

            return labelsets, None


def merge_overlapping_labels():

    pass
    # get the overlap (voxel count) of labels within the watershed object
    # counts = np.bincount(imregion[labelmask])
    # if len(counts) > 1:  # continue if there is anything else than zero's in the region
    #     # select the largest candidate overlapping the watershed
    #     candidate = np.argmax(counts[1:]) + 1
    #     # TODO: improve criteria for accepting candidate
    #     # only select it if the overlap is larger than min_labelsize
    #     if ((counts[candidate] > min_labelsize) and
    #             (candidate != prop.label)):
    #         # TODO: pop label+direction from list if merged?
    #         print('merging {} and {}'.format(prop.label, candidate))
    #         labelset = set([prop.label, candidate])
    #         labelsets = utils.classify_label_set(labelsets, labelset, prop.label)
    #         # TODO: blocky fills when not enough space around the label for sufficient negative label
    #         # TODO: include boundary as possible candidate


def split_label(image):

    labels_split = label(image)

    try:
        rp = regionprops(labels_split)  # TypeError: Only 2-D and 3-D images supported.
        # TODO: identify NoR
        # e.g. project centreline, do slicewise labeling over the gap and see 
        # if the label that includes the centreline touched the boundary of the slice/cylinder
        # to identify the z-range of the NoR (if detected)
        return labels_split, rp
    except TypeError:  # TODO: delete single-voxel labels?
        return None, None


def check_split_label(prop, mo, checkonly=False):
    
    labels_split, rp = split_label(prop.image)
    
    if rp is None:
        print('single voxel: {}'.format(prop.label))
        delete_single_voxel(prop, mo)
    elif len(rp) == 1:
        pass
    else:
        print('split label: {} {}'.format(prop.label, len(rp)))
        if checkonly:
            for p in rp:
                print(p.bbox, p.area, prop.centroid)
        else:
            correct_split_label(prop, mo, rp, labels_split)


def delete_single_voxel(prop, mo):

    z, y, x, Z, Y, X = prop.bbox
    dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
    mo.slices = mo.get_slice_objects(dataslices)
    out = mo.slice_dataset(squeeze=False)

    mask = out == prop.label
    out[mask] = 0

    mo.write(out)


def correct_split_label(prop, mo, rp, labels_split):

    z, y, x, Z, Y, X = prop.bbox
    dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
    mo.slices = mo.get_slice_objects(dataslices)
    out = mo.slice_dataset(squeeze=False)

    out, mo.maxlabel = split_label_corr(labels_split, rp, mo.maxlabel, out)

    mo.write(out)


def split_label_corr(labels_split, rp, maxlabel, out):

    rp = sort_rp_on_size(rp)
    for p in rp[1:]:
        mask = labels_split == p.label
        if p.area > 10:
            maxlabel += 1
            newval = maxlabel
        else:
            newval = 0
        out[mask] = newval
#         print(p.label, maxlabel, p.area, newval)

    return out, maxlabel


def connect_split_label(prop, im, mo, data, mask, searchradius, axons=None):

    _, rp = split_label(prop.image)

    if rp is None:
        return

    if len(rp) >= 2:

        rp = sort_rp_on_z(rp)

        for i, prop_bot in enumerate(rp[:-1]):
            prop_top = rp[i+1]
            wsout = connect_split_label_pair(im, prop, prop_bot, prop_top, searchradius, data, mask, mo, axons)
            if wsout is not None:
                mo.write(wsout)


def connect_split_label_pair(im, prop, prop_bot, prop_top, searchradius, data, mask, mo, axons):

    slices = get_region_slices_between(im, prop, prop_bot, prop_top, searchradius)[0]

    if slices[0].start >= slices[0].stop:  # FIXME
        return

    data.slices = mask.slices = mo.slices = slices
    if axons is not None:
        axons.slices = slices

    wsout = fill_between_labels(prop, mo, data, mask, axons)

    return wsout


def fill_between_labels(prop, mo, data, mask, axons):

    im_ds = mo.slice_dataset(squeeze=False)
    data_ds = data.slice_dataset(squeeze=False)
    mask_ds = mask.slice_dataset(squeeze=False)
    if axons is not None:
        axons_ds = axons.slice_dataset(squeeze=False)
    else:
        axons_ds = None

    mask = ~mask_ds.astype('bool')

    seeds = create_seeds(im_ds, prop.label, prop.label, axons_ds, cylinder=1)

    ws = watershed(-data_ds, seeds, mask=mask)

    mask_label = ws == 1
    wsout = im_ds
    wsout[mask_label] = prop.label

#     # make sure resulting label is contiguous; split if not
#     labels_split, rp = split_label(mask_label)
#     if rp is None:
#         print('single voxel: {}'.format(prop.label))
#     elif len(rp) == 1:
#         pass
#     else:
#         print('split label: {} {}'.format(prop.label, len(rp)))
#         wsout, mo.maxlabel = split_label_corr(labels_split, rp, mo.maxlabel, wsout)

    return wsout


def detect_NoR(seeds, seedlabel, maskMM):

    slc_idxs = []
    for slc_idx in range(1, seeds.shape[0] - 1):
        seeds_slc = seeds[slc_idx, :, :]
        maskMM_slc = maskMM[slc_idx, :, :]
        labels = label(maskMM_slc)
        seedsmask = seeds_slc == seedlabel
        ulabels = set(labels[seedsmask].ravel()) - set([0])
        mask_cyl = seeds_slc != -1
        ulabels_outside = set(labels[~mask_cyl].ravel()) - set([0])
        if ulabels:
            # print(ulabels, ulabels_outside)
            if list(ulabels)[0] in ulabels_outside:
                # slc_idx does not have closed sheath
                slc_idxs.append(slc_idx)

    return slc_idxs


def detect_NoR_from_labels(prop):

    slc_idxs = []
    for i, (slc, slc_aux) in enumerate(zip(prop.image, prop.intensity_image)):
        rim = np.logical_xor(binary_erosion(slc, border_value=1), slc)
        if np.sum(slc_aux[rim]) == len(slc_aux[rim]):
            slc_idxs.append(i)

    return slc_idxs


def connect_to_borders(prop, im, mo, data=None, maskMM=None,
                       searchradius=[100, 30, 30], maskDS=None):

    for direction in ['down', 'up']:

        slices = get_region_slices_projected(direction, im, prop,
                                             searchradius,
                                             z_extent=10)[0]  # z_extent=searchradius[0]

        # print(prop.label, direction, slices)

        # continue if borderslice is already top/bottom
        if (direction == 'down' and (slices[0].stop == 1) or
                direction == 'up' and (slices[0].start == mo.dims[0] - 1)):  # i.e. b_idx == dims[0] - 1
            # print("continue if borderslice is already top/bottom")
            continue
        # continue if dataset border is outside search region
        if ((direction == 'down' and (slices[0].start != 0)) or
                (direction == 'up' and (slices[0].stop != mo.dims[0]))):
            # print("continue if dataset border is outside search region")
            continue

        mo.slices = data.slices = slices
        labels_ds = mo.slice_dataset(squeeze=False)
        data_ds = data.slice_dataset(squeeze=False)
        if maskMM is not None:
            maskMM.slices = slices
            maskMM_ds = maskMM.slice_dataset(squeeze=False)
            # mask = binary_dilation(~maskMM_ds.astype('bool'))
            mask = ~maskMM_ds.astype('bool')
        else:
            mask = np.ones_like(mo.ds[:], dtype='bool')
        if maskDS is not None:
            maskDS.slices = slices
            maskDS_ds = maskDS.slice_dataset(squeeze=False)
        else:
            maskDS_ds = None

        if len(np.unique(labels_ds)) < 2:
            # print("len(np.unique(labels_ds)) < 2")
            continue  # label 0 and prop.label assumed to be there

        seeds = create_seeds(labels_ds, prop.label, prop.label, cylinder=2, labval=prop.label)  # cylinder maybe too big
        if maskMM is not None:
            ws = watershed(-data_ds, seeds, mask=mask)
        else:
            ws = watershed(-data_ds, seeds)

        mask_label = ws == prop.label
        wsout = labels_ds
        wsout[mask_label] = prop.label
        mo.write(wsout)


def fill_to_borders(prop, mo, data, mask, axons):

    im_ds = mo.slice_dataset(squeeze=False)
    data_ds = data.slice_dataset(squeeze=False)
    mask_ds = mask.slice_dataset(squeeze=False)
    if axons is not None:
        axons_ds = axons.slice_dataset(squeeze=False)
    else:
        axons_ds = None

    seeds = create_seeds(im_ds, prop.label, prop.label, axons_ds, cylinder=1)
    mask = ~mask_ds.astype('bool')
    # FIXME: add cylinder to mask instead:
    # maskcyl = create_seeds(im_ds, prop.label, prop.label, axons_ds, cylinder=1)
    # mask = maskcyl != -1
    # mask = np.logical_and(~mask_ds.astype('bool'), mask)
    ws = watershed(-data_ds, seeds, mask=mask)

    mask_label = ws == 1
    wsout = im_ds
    wsout[mask_label] = prop.label

    return wsout


def fill_connected_labels(image_in,
                          data_in='',
                          maskMM_in='',
                          searchradius=[10, 10, 10],
                          check_split=False, checkonly=False,
                          between=False, to_border=False,
                          outputpath='',
                          usempi=False):

    mpi = wmeMPI(usempi)

    svoxs = utils.get_image(image_in, imtype='Label', comm=mpi.comm)

    if data_in:
        data = utils.get_image(data_in, comm=mpi.comm)
 
    if maskMM_in:
        mask = utils.get_image(maskMM_in, imtype='Mask', comm=mpi.comm)
        maskds = mask.ds
    else:
        mask = None
        maskds = None

    props = svoxs.get_props(protective=False)
    mo = LabelImage(outputpath, **props)
    mo.create(comm=mpi.comm)

    mo.ds[:] = np.copy(svoxs.ds[:])
    mo.set_maxlabel()

    rp_main = regionprops(svoxs.ds, maskds)
    mpi.nblocks = len(rp_main)
    mpi.scatter_series()  # randomize=True

    for i in mpi.series:
        prop = rp_main[i]

        if between:
            connect_split_label(prop, svoxs, mo, data, mask, searchradius, axons=None)
            
        if to_border:
            connect_to_borders(prop, svoxs, mo, data, mask, searchradius, axons=None)
        if check_split:
            check_split_label(prop, mo, checkonly=checkonly)

    mo.close()
    svoxs.close()
    if data_in:
        data.close()
    if maskMM_in:
        mask.close()


def create_seeds(imregion, label_top, label_bot,
                 axons=None, cylinder=0, labval=1):

    seeds = np.zeros_like(imregion, dtype='int')

    # find the mask of the label in the seedplane
    mask_bot = imregion[0, :, :] == label_bot
    mask_top = imregion[-1, :, :] == label_top

    # label the surround with -1 (and the area to fill with 0)
    if cylinder:
        seeds = create_seeds_cylinder(seeds, mask_bot, mask_top, cylinder)
        # TODO: is this creating the correct seed in the borderslice
        # or does it only do the cylinder-surround?
    else:
        # only fill with surround if label is in the top/bot slice
        if np.sum(mask_bot) != 0:
            seeds[0, :, :][~mask_bot] = -1
        if np.sum(mask_top) != 0:
            seeds[-1, :, :][~mask_top] = -1

    # label the seeds with 1 (or label value)
    seeds[seeds == 1] = labval
    seeds[0, :, :][mask_bot] = labval
    seeds[-1, :, :][mask_top] = labval

    if axons is not None:
        mask_axons = axons == label_top
        mask_fill_area = seeds == 0
        mask = np.logical_and(mask_axons, mask_fill_area)
        seeds[mask] = labval

    return seeds


def create_seeds_cylinder(seeds, mask_bot=None, mask_top=None,
                          cylinder_factor=1,
                          fill_centerline=False):

    bot = np.sum(mask_bot) == 0
    top = np.sum(mask_top) == 0

    if bot and top:
        return seeds

    if ~bot and ~top:
        ctr_bot, ed_bot = get_centroid(mask_bot)
        ctr_top, ed_top = get_centroid(mask_top)

    if ~bot and top:
        ctr_bot, ed_bot = get_centroid(mask_bot)
        ctr_top, ed_top = ctr_bot, ed_bot

    if bot and ~top:
        ctr_top, ed_top = get_centroid(mask_top)
        ctr_bot, ed_bot = ctr_top, ed_top

    disk_radius = max(ed_bot, ed_top) * cylinder_factor
    if disk_radius < 5:
        disk_radius = 5
    disk_ed = disk(int(disk_radius))

    # project line from one centroid to the other
    n_slices = seeds.shape[0]
    xs = np.linspace(ctr_bot[1], ctr_top[1], n_slices)
    ys = np.linspace(ctr_bot[0], ctr_top[0], n_slices)

    # calculate expected centroid voxel for each slice
    start = 1 if ~bot else 0
    stop = seeds.shape[0] - 1 if ~top else seeds.shape[0]
    for slc in range(start, stop):
        seeds_slc = np.zeros_like(seeds[slc, :, :])
        seeds_slc[int(ys[slc]), int(xs[slc])] = 1
        seeds_slc = binary_dilation(seeds_slc, disk_ed).astype('int')

        if (~bot and ~top) or fill_centerline:
            # TODO: only do this for short ranges??
            seeds_slc[int(ys[slc]), int(xs[slc])] = 2

        seeds[slc, :, :] = seeds_slc

    seeds -= 1

    return seeds


def sort_rp_on_z(rp):

    z_idxs = []
    for p in rp:
        z_idxs.append(p.bbox[0])
    rp_idxs = np.argsort(z_idxs)

    return [rp[i] for i in rp_idxs]


def sort_rp_on_size(rp):

    sizes = []
    for p in rp:
        sizes.append(p.area)
    rp_idxs = np.argsort(-np.array(sizes))

    return [rp[i] for i in rp_idxs]


def get_borderslice(direction, prop):
    """Return the index of the top/bottom section of a label."""

    if direction == 'down':  # a box below the label bbox
        borderslice = int(prop.bbox[0])
    elif direction == 'up':  # a box above the label bbox
        borderslice = int(prop.bbox[3]) - 1

    return borderslice


def get_centroid(mask):

    rp = regionprops(mask.astype('i'))
    centroid = rp[0].centroid
    eq_diam = rp[0].equivalent_diameter

    return centroid, eq_diam


def get_zZ(direction, im, prop, searchradius=[20, 20, 20]):
    """Return the z-range of a box above/below the label's bbox."""

    if direction == 'down':  # a box below the label bbox including borderslice
        b_idx = get_borderslice(direction, prop)
        z = max(0, b_idx - searchradius[0])
        Z = b_idx + 1
    elif direction == 'up':  # a box above the label bbox including borderslice
        b_idx = get_borderslice(direction, prop)
        z = b_idx
        Z = min(im.dims[0], b_idx + searchradius[0] + 1)

    return z, Z


def get_xXyY_from_centroid(im, centroid, searchradius=[20, 20, 20]):

    # get the x,y-range of a box above/below the label's bbox
    y = max(0, int(centroid[0]) - searchradius[1])
    Y = min(im.dims[1], int(centroid[0]) + searchradius[1] + 1)
    x = max(0, int(centroid[1]) - searchradius[2])
    X = min(im.dims[2], int(centroid[1]) + searchradius[2] + 1)

    return x, X, y, Y


def get_xXyY_from_bbox(im, prop, searchradius=[20, 20, 20]):

    # get the x,y-range of a box above/below the label's bbox
    y = max(0, int(prop.bbox[1]) - searchradius[1])
    Y = min(im.dims[1], int(prop.bbox[4]) + searchradius[1])
    x = max(0, int(prop.bbox[2]) - searchradius[2])
    X = min(im.dims[2], int(prop.bbox[5]) + searchradius[2])

    return x, X, y, Y


def get_region_slices(direction, im, prop, searchradius=[20, 20, 20]):
    """Find coordinates of a box bordering a partial label."""

    """NOTE:
    prop.bbox is in half-open interval
    """

    z, Z = get_zZ(direction, im, prop, searchradius)

    b_idx = get_borderslice(direction, prop)
    mask_label = im.ds[b_idx, :, :] == prop.label
    centroid, _ = get_centroid(mask_label)
    x, X, y, Y = get_xXyY_from_centroid(im, centroid, searchradius)

    dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
    slices = im.get_slice_objects(dataslices)

    return slices, (x, X, y, Y, z, Z)


def get_region_slices_around(im, prop, searchradius=[20, 20, 20]):
    """Find coordinates of a box around a label."""

    z = max(0, int(prop.bbox[0]) - searchradius[0])
    Z = min(im.dims[0], int(prop.bbox[3]) + searchradius[0] + 1)
    x, X, y, Y = get_xXyY_from_bbox(im, prop, searchradius)

    dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
    slices = im.get_slice_objects(dataslices)

    return slices, (x, X, y, Y, z, Z)


def get_region_slices_between(im, prop, bot, top, searchradius=[20, 20, 20]):
    """Find coordinates of a box in between two labels."""

    z = int(prop.bbox[0]) + int(bot.bbox[3]) - 1
    Z = int(prop.bbox[0]) + int(top.bbox[0]) + 1
    x, X, y, Y = get_xXyY_from_bbox(im, prop, searchradius)

    dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
    slices = im.get_slice_objects(dataslices)

    return slices, (x, X, y, Y, z, Z)


def get_region_slices_projected(direction, im, prop,
                                searchradius=[20, 20, 20], z_extent=10):

    # centroids of borderslice and labelslice (i.e. the one *z_extent* sections down/up)
    b_idx = get_borderslice(direction, prop)  # border section index
    z, Z = get_zZ(direction, im, prop, searchradius)  # z and Z for start, stop in slice
    l_idx = {'down': b_idx + z_extent,
             'up': b_idx - z_extent}[direction]

#     print(z_extent, b_idx, l_idx, z, Z)
    if (l_idx < 0
            or l_idx > (im.dims[0] - 1)):
        # or (Z-z < z_extent)
        # FIXME: the final condition only valid if Z-z pertains to box around label
        print('no projection: section at z_extent out of bounds')
        return get_region_slices(direction, im, prop, searchradius)

    b_mask = im.ds[b_idx, :, :] == prop.label
    l_mask = im.ds[l_idx, :, :] == prop.label

    if np.sum(b_mask) == 0 or np.sum(l_mask) == 0:
        print('no projection: bordersection or section at z_extent does not contain prop.label')
        return get_region_slices(direction, im, prop, searchradius)

    # print(np.sum(b_mask), np.sum(l_mask))
    b_ctr, _ = get_centroid(b_mask)  # coords in full image
    l_ctr, _ = get_centroid(l_mask)

    # project line to z - searchradius[0] or Z + searchradius[0]
    fac = searchradius[0] / z_extent
    b_point = [b_idx, b_ctr[0], b_ctr[1]]
    l_point = [l_idx, l_ctr[0], l_ctr[1]]
    p_point = [c0 + (c0-c1) * fac for c0, c1 in zip(b_point, l_point)]
    # print(l_point, b_point, p_point)
    p_ctr = [p_point[1], p_point[2]]

    # get the z-range
    if direction == 'up':
        z = b_idx
        Z = min(int(p_point[0]), im.dims[0])
    elif direction == 'down':
        z = max(0, int(p_point[0]))
        Z = b_idx + 1

    # get the xy-range for the borderslice and projected point
    x1, X1, y1, Y1 = get_xXyY_from_centroid(im, b_ctr, searchradius)
    x2, X2, y2, Y2 = get_xXyY_from_centroid(im, p_ctr, searchradius)
    # combine ranges
    y = min(y1, y2)
    Y = max(Y1, Y2)
    x = min(x1, x2)
    X = max(X1, X2)

    # dataslices = [z, Z+1, 1, y, Y+1, 1, x, X+1, 1]  # is this half-open?? moved the +1 to 'down only'
    dataslices = [z, Z, 1, y, Y, 1, x, X, 1]
    slices = im.get_slice_objects(dataslices)

    # TODO: cylinder
    # # project line from one centroid to the other
    # n_slices = z_extent + searchradius[0]
    # xs = np.linspace(y1, y2, n_slices)
    # ys = np.linspace(x1, x2, n_slices)

    return slices, (x, X, y, Y, z, Z)


def create_mask(outputpath, image_in, mask_in, mask_ds=''):

    tv = utils.get_image(image_in, imtype='Label')
    maskWS = utils.get_image(mask_in, imtype='Mask')
    if mask_ds:
        maskDS = utils.get_image(mask_ds, imtype='Mask')

    props = maskWS.get_props()
    mo = MaskImage(outputpath, **props)
    mo.create()

    if mask_ds:
        mask = maskDS.ds[:, :, :]
        mask[tv.ds[:,:,:] != 0] = 0
        mask[maskWS.ds[:,:,:] == 1] = 0
    else:
        mask = maskWS.ds[:, :, :]
        mask[tv.ds[:,:,:] != 0] = 0

    mo.write(mask)

    mo.close()
    tv.close()
    maskWS.close()
    if mask_ds:
        maskDS.close()


if __name__ == "__main__":
    main(sys.argv[1:])
