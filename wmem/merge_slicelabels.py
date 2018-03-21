#!/usr/bin/env python

"""Merge slice-wise overlapping labels.

"""

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
            args.usempi & ('mpi4py' in sys.modules),
            args.outputfile,
            args.protective,
            )

    elif args.mode == "MAfwmap":

        map_labels(
            args.inputfile,
            args.maskMM,
            args.min_labelsize,
            args.close,
            args.relabel,
            args.outputfile,
            )


def evaluate_overlaps(
        h5path_in,
        slicedim,
        offsets,
        threshold_overlap,
        usempi=False,
        outputfile='',
        protective=False,
        ):
    """Check for slicewise overlaps between labels."""

    # open data for reading
    h5file_in, ds_in, _, _ = utils.h5_load(h5path_in)

    # prepare mpi  # TODO: could allow selection of slices/subset here
    mpi_info = utils.get_mpi_info(usempi)
    n_slices = ds_in.shape[slicedim] - offsets
    series = np.array(range(0, n_slices), dtype=int)
    if mpi_info['enabled']:
        series = utils.scatter_series(mpi_info, series)[0]

    # merge overlapping neighbours
    MAlist = []
    for i in series:
        print("processing slice %d" % i)
        for j in range(1, offsets):

            data_section = utils.get_slice(ds_in, i, slicedim)
            nb_section = utils.get_slice(ds_in, i+j, slicedim)

            MAlist = merge_neighbours(MAlist,
                                      data_section, nb_section,
                                      threshold_overlap)

    h5file_in.close()

    # dump the list of overlapping neighbours in a pickle
    root = outputfile.split('.h5')[0]
    mname = "_host-{}_rank-{:02d}.pickle".format(socket.gethostname(),
                                                 mpi_info['rank'])
    with open(root + mname, "wb") as f:
        pickle.dump(MAlist, f)

    # wait for all processes to finish
    if mpi_info['enabled']:
        mpi_info['comm'].Barrier()

    # let one process combine the overlaps found in the separate processes
    if mpi_info['rank'] == 0:
        match = root + "_host*_rank*.pickle"
        infiles = glob.glob(match)
        for ppath in infiles:
            with open(ppath, "r") as f:
                newlist = pickle.load(f)
            for labelset in newlist:
                MAlist = utils.classify_label_list(MAlist, labelset)

        with open(root + ".pickle", "wb") as f:
            pickle.dump(MAlist, f)


def map_labels(
        h5path_in,
        h5path_mm='',
        min_labelsize=0,
        close=None,
        relabel=False,
        h5path_out='',
        protective=False,
        ):
    """Map groups of labels to a single label."""

    # check output path
    if '.h5' in h5path_out:
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # load the pickled list of neighbours
    root = h5path_out.split('.h5')[0]
    with open(root + ".pickle", "r") as f:
        MAlist = pickle.load(f)

    # apply forward map
    ulabels = np.unique(ds_in[:])
    fw = [l if l in ulabels else 0
          for l in range(0, np.amax(ulabels) + 1)]
    labels = utils.forward_map_list(np.array(fw), ds_in[:], MAlist)

    if min_labelsize:
        remove_small_objects(labels, min_size=min_labelsize, in_place=True)

    if close is not None:
        if len(close) == 1:
            selem = ball(close)
        elif len(close) == 3:
            selem = generate_anisotropic_selem(close)
        labels = closing(labels, selem)

    if h5path_mm:
        h5file_mm, ds_mm, _, _ = utils.h5_load(h5path_mm)
        labels[ds_mm[:].astype('bool')] = 0
        h5file_mm.close()

    if min_labelsize:
        remove_small_objects(labels, min_size=min_labelsize, in_place=True)

    if relabel:
        labels = relabel_sequential(labels)[0]

    ds_out[:, :, :] = labels

    # close and return
    try:
        h5file_in.close()
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


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


def merge_neighbours(MAlist, data_section, nb_section, threshold_overlap=0.01):
    """Adapt the forward map to merge neighbouring labels."""

    data_labels = np.trim_zeros(np.unique(data_section))
    for data_label in data_labels:

        mask_data = data_section == data_label
        bins = np.bincount(nb_section[mask_data])
        if len(bins) <= 1:
            continue

        nb_label = np.argmax(bins[1:]) + 1
        if nb_label == data_label:
            continue
        n_data = np.sum(mask_data)
        n_nb = bins[nb_label]
        if float(n_nb) / float(n_data) < threshold_overlap:
            continue

        MAlist = utils.classify_label_list(MAlist, set([data_label, nb_label]))

    return MAlist


if __name__ == "__main__":
    main(sys.argv[1:])
