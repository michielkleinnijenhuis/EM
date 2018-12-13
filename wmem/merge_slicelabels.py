#!/usr/bin/env python

"""Merge slice-wise overlapping labels.

"""

import os
import sys
import argparse
import pickle
import glob
import socket

import numpy as np
from skimage.morphology import remove_small_objects, closing, ball, disk
from skimage.segmentation import relabel_sequential
try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")

from wmem import parse, utils


def main(argv):
    """Merge slice-wise overlapping labels."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_merge_slicelabels(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    if args.mode == "MAstitch":

        evaluate_overlaps(
            args.inputfile,
            args.slicedim,
            args.offsets,
            args.threshold_overlap,
            args.do_map_labels,
            args.maskMM,
            args.min_labelsize,
            args.close,
            args.relabel_from,
            args.usempi & ('mpi4py' in sys.modules),
            args.outputfile,
            args.save_steps,
            args.protective,
            )

    elif args.mode == "MAfwmap":

        map_labels(
            args.inputfile,
            args.outputfile,
            args.save_steps,
            args.protective,
            )

    elif args.mode == "MAfilter":

        filter_labels(
            args.inputfile,
            args.maskMM,
            args.min_labelsize,
            args.close,
            args.relabel_from,
            args.outputfile,
            args.save_steps,
            args.protective,
            )


def evaluate_overlaps(
        h5path_in,
        slicedim,
        offsets,
        threshold_overlap,
        do_map_labels=False,
        h5path_mm='',
        min_labelsize=0,
        close=None,
        relabel_from=0,
        usempi=False,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Check for slicewise overlaps between labels."""

    # prepare mpi  # TODO: could allow selection of slices/subset here
    mpi_info = utils.get_mpi_info(usempi)

    # open data for reading
    h5file_in, ds_in, _, _ = utils.h5_load(h5path_in, comm=mpi_info['comm'])

    n_slices = ds_in.shape[slicedim] - offsets
    series = np.array(range(0, n_slices), dtype=int)
    if mpi_info['enabled']:
        series = utils.scatter_series(mpi_info, series)[0]

    # merge overlapping neighbours
    labelsets = {}
    for i in series:
        print("processing slice {}".format(i))
        for j in range(1, offsets):

            data_section = utils.get_slice(ds_in, i, slicedim)
            nb_section = utils.get_slice(ds_in, i+j, slicedim)

            labelsets = merge_neighbours(labelsets,
                                         data_section, nb_section,
                                         threshold_overlap)

    # dump the list of overlapping neighbours in a pickle
    h5root = h5file_in.filename.split('.h5')[0]
    ds_out_name = os.path.split(h5path_out)[1]
    mname = "host-{}_rank-{:02d}".format(socket.gethostname(), mpi_info['rank'])
    lsroot = '{}_{}_{}'.format(h5root, ds_out_name, mname)
    utils.write_labelsets(labelsets, lsroot, ['pickle'])

    h5file_in.close()

    # wait for all processes to finish
    if mpi_info['enabled']:
        mpi_info['comm'].Barrier()

    # let one process combine the overlaps found in the separate processes
    if mpi_info['rank'] == 0:
        lsroot = '{}_{}'.format(h5root, ds_out_name)
        match = "{}_host*_rank*.pickle".format(lsroot)
        infiles = glob.glob(match)
        for ppath in infiles:
            with open(ppath, "r") as f:
                newlabelsets = pickle.load(f)
            for lsk, lsv in newlabelsets.items():
                labelsets = utils.classify_label_set(labelsets, lsv, lsk)

        utils.write_labelsets(labelsets, lsroot, ['txt', 'pickle'])

        if do_map_labels:
            map_labels(h5path_in, h5path_mm,
                       min_labelsize, close, relabel_from,
                       h5path_out, save_steps, protective)


def map_labels(
        h5path_in,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Map groups of labels to a single label."""

    # check output path
    outpaths = {'out': h5path_out, 'stitched': ''}
    root, ds_main = outpaths['out'].split('.h5')
    for dsname, outpath in outpaths.items():
        grpname = ds_main + "_steps"
        outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # load the pickled set of neighbours
    lsroot = h5path_out.split('.h5')[0]
    lspath = '{}_{}.pickle'.format(lsroot, ds_out.name[1:])
    with open(lspath, "r") as f:
        labelsets = pickle.load(f)

    # apply forward map
    ulabels = np.unique(ds_in[:])
    maxlabel = np.amax(ulabels)
    fw = [l if l in ulabels else 0 for l in range(0, maxlabel + 1)]
    labels = utils.forward_map(np.array(fw), ds_in[:], labelsets)

    if save_steps:
        fw = np.zeros(maxlabel + 1, dtype='i')
        MAlabels = utils.forward_map(np.array(fw), ds_in[:], labelsets)
        utils.save_step(outpaths, 'stitched', MAlabels, elsize, axlab)

    ds_out[:, :, :] = labels

    # close and return
    try:
        h5file_in.close()
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def filter_labels(
        h5path_in,
        h5path_mm='',
        min_labelsize=0,
        close=None,
        relabel_from=0,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Map groups of labels to a single label."""

    # check output path
    outpaths = {'out': h5path_out, 'closed': ''}
    root, ds_main = outpaths['out'].split('.h5')
    for dsname, outpath in outpaths.items():
        grpname = ds_main + "_steps"
        outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    labels = ds_in[:]

#     if min_labelsize:
#         remove_small_objects(labels, min_size=min_labelsize, in_place=True)

    if close is not None:
        labels = close_labels(labels, close)
        if h5path_mm:
            print('removing voxels in mask')
            h5file_mm, ds_mm, _, _ = utils.h5_load(h5path_mm)
            labels[ds_mm[:].astype('bool')] = 0
            h5file_mm.close()
        if save_steps:
            utils.save_step(outpaths, 'closed', labels, elsize, axlab)

    if min_labelsize:
        print('removing small labels')
        remove_small_objects(labels, min_size=min_labelsize, in_place=True)
#         if save_steps:
#             utils.save_step(outpaths, 'small', smalllabels, elsize, axlab)

    if relabel_from > 1:
        print('relabeling from {}'.format(relabel_from))
        labels = relabel_sequential(labels, relabel_from)[0]
        # TODO: save mapping?

    ds_out[:, :, :] = labels

    # close and return
    try:
        h5file_in.close()
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def close_labels(labels, close):
    """Apply forward map."""

    print('closing labels')
    if len(close) == 1:
        selem = ball(close)
    elif len(close) == 3:
        selem = generate_anisotropic_selem(close)
    labels = closing(labels, selem)

    return labels


def generate_anisotropic_selem(close):
    """Generate an anisotropic diamond structuring element."""

    # FIXME: for now assumed z is first dimension
    # and x,y are in-plane with the same width
    selem_dim = [2*close[0]+1, 2*close[1]+1, 2*close[2]+1]
    selem = np.zeros(selem_dim)
    for i in range(close[0] + 1):
        xywidth = max(0, close[1] - (close[0] - i))
        xy = disk(xywidth)
        padsize = (selem_dim[1] - len(xy)) / 2
        xy = np.pad(xy, padsize, 'constant')
        selem[i, :, :] = xy
        selem[-(i+1), :, :] = xy

    return selem


def merge_neighbours(labelsets, data_section, nb_section, threshold_overlap=0.01):
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
        dice = np.sum(np.logical_and(mask_nb, mask_data)) * 2.0 / (np.sum(mask_nb) + n_data)
#         n_nb = bins[nb_label]
#         if float(n_nb) / float(n_data) < threshold_overlap:
        if dice < threshold_overlap:
            continue

#         print(dice)
        labelsets = utils.classify_label_set(labelsets, set([data_label, nb_label]))

    return labelsets


if __name__ == "__main__":
    main(sys.argv[1:])
