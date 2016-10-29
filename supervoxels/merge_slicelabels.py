#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
import pickle
import glob

from skimage.morphology import remove_small_objects, closing, ball

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', nargs=2,
                        default=['_labelMA', '/stack'],
                        help='...')
    parser.add_argument('--maskMM', nargs=2, default=None,
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMA_core2D_merged', 'stack'],
                        help='...')
    parser.add_argument('-d', '--slicedim', type=int, default=0,
                        help='...')

    parser.add_argument('-M', '--mode',
                        help='...')

    parser.add_argument('-r', '--offsets', default=4, type=int,
                        help='...')
    parser.add_argument('-t', '--threshold_overlap', default=0.01, type=float,
                        help='...')
    parser.add_argument('-s', '--min_labelsize', default=10000, type=int,
                        help='...')
    parser.add_argument('-c', '--close', default=None, type=int,
                        help='...')

    parser.add_argument('-m', '--usempi', action='store_true',
                        help='use mpi4py')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    maskMM = args.maskMM
    outpf = args.outpf
    slicedim = args.slicedim
    mode = args.mode
    offsets = args.offsets
    threshold_overlap = args.threshold_overlap
    min_labelsize = args.min_labelsize
    close = args.close
    usempi = args.usempi & ('mpi4py' in sys.modules)

    if mode == "MAstitch":

        evaluate_overlaps(datadir, dset_name, labelvolume, slicedim,
                          offsets, threshold_overlap, usempi, outpf)

    elif mode == "MAfwmap":

        map_labels(datadir, dset_name, labelvolume, maskMM,
                   min_labelsize, close, outpf)


# ========================================================================== #
# function defs
# ========================================================================== #


def scatter_series(n, comm, size, rank, SLL):
    """Scatter a series of jobnrs over processes."""

    nrs = np.array(range(0, n), dtype=int)
    local_n = np.ones(size, dtype=int) * n / size
    local_n[0:n % size] += 1
    local_nrs = np.zeros(local_n[rank], dtype=int)
    displacements = tuple(sum(local_n[0:r]) for r in range(0, size))
    comm.Scatterv([nrs, tuple(local_n), displacements,
                   SLL], local_nrs, root=0)

    return local_nrs, tuple(local_n), displacements


def evaluate_overlaps(datadir, dset_name, labelvolume, slicedim,
                      offsets, threshold_overlap,
                      usempi, outpf):
    """Check for slicewise overlaps between labels."""

    fname = os.path.join(datadir, dset_name + labelvolume[0] + '.h5')

    if usempi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        f = h5py.File(fname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        fstack = f[labelvolume[1]]

        n_slices = fstack.shape[slicedim] - offsets
        local_nrs = scatter_series(n_slices, comm, size, rank,
                                   MPI.SIGNED_LONG_LONG)[0]
    else:
        f = h5py.File(fname, 'r')
        fstack = f[labelvolume[1]]

        n_slices = fstack.shape[slicedim] - offsets
        local_nrs = np.array(range(0, n_slices), dtype=int)

    MAlist = []
    for i in local_nrs:
        print("processing slice %d" % i)
        for j in range(1, offsets):

            if slicedim == 0:
                data_section = fstack[i, :, :]
                nb_section = fstack[i+j, :, :]
            elif slicedim == 1:
                data_section = fstack[:, i, :]
                nb_section = fstack[:, i+j, :]
            elif slicedim == 2:
                data_section = fstack[:, :, i]
                nb_section = fstack[:, :, i+j]

            MAlist = merge_neighbours(MAlist,
                                      data_section, nb_section,
                                      threshold_overlap)

    f.close()

    filename = dset_name + outpf + "_rank%04d.pickle" % rank
    filepath = os.path.join(datadir, filename)
    with open(filepath, "wb") as file:
        pickle.dump(MAlist, file)

    comm.Barrier()

    if rank == 0:
        match = dset_name + outpf + "_rank*.pickle"
        infiles = glob.glob(os.path.join(datadir, match))
        for filepath in infiles:
            with open(filepath, "r") as file:
                newlist = pickle.load(file)
            for labelset in newlist:
                MAlist = classify_label(MAlist, labelset)

        filename = dset_name + outpf + ".pickle"
        filepath = os.path.join(datadir, filename)
        with open(filepath, "wb") as file:
            pickle.dump(MAlist, file)


def map_labels(datadir, dset_name, labelvolume, maskMM,
               min_labelsize, close, outpf):
    """Map groups of labels to a single label."""

    filename = dset_name + outpf[0] + ".pickle"
    filepath = os.path.join(datadir, filename)
    with open(filepath, "r") as file:
        MAlist = pickle.load(file)

    fname = os.path.join(datadir, dset_name + labelvolume[0] + '.h5')
    f = h5py.File(fname, 'r')
    fstack = f[labelvolume[1]]

    ulabels = np.unique(fstack)
    fw = [l if l in ulabels else 0
          for l in range(0, np.amax(ulabels) + 1)]
    labels = forward_map(np.array(fw), fstack, MAlist)

    remove_small_objects(labels, min_size=min_labelsize, in_place=True)

    if closing is not None:
        labels = closing(labels, ball(close))

    if maskMM is not None:
        mname = os.path.join(datadir, dset_name + maskMM[0] + '.h5')
        m = h5py.File(mname, 'r')
        labels[np.array(m[maskMM[1]], dtype='bool')] = 0
        m.close()

    remove_small_objects(labels, min_size=min_labelsize, in_place=True)

    gname = os.path.join(datadir, dset_name + outpf[0] + '.h5')
    g = h5py.File(gname, 'w')
    outds = g.create_dataset(outpf[1], fstack.shape,
                             dtype=fstack.dtype,
                             compression='gzip')
    outds[:, :, :] = labels
    elsize, al = get_h5_attributes(fstack)
    write_h5_attributes(outds, elsize, al)

    g.close()
    f.close()


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

        MAlist = classify_label(MAlist, set([data_label, nb_label]))

    return MAlist


def classify_label(MAlist, labelset):
    """Add set of labels to an axonset or create new axonset."""

    found = False
    for i, MA in enumerate(MAlist):
        for l in labelset:
            if l in MA:
                MAlist[i] = MA | labelset
                found = True
                break
    if not found:
        MAlist.append(labelset)

    return MAlist


def forward_map(fw, labels, MAlist):
    """Map all labelsets in MAlist to axons."""

    for MA in MAlist:
        MA = sorted(list(MA))
        for l in MA:
            fw[l] = MA[0]

    fwmapped = fw[labels]

    return fwmapped


def get_h5_attributes(stack):
    """Get attributes from a stack."""

    element_size_um = axislabels = None

    if 'element_size_um' in stack.attrs.keys():
        element_size_um = stack.attrs['element_size_um']

    if 'DIMENSION_LABELS' in stack.attrs.keys():
        axislabels = stack.attrs['DIMENSION_LABELS']

    return element_size_um, axislabels


def write_h5_attributes(stack, element_size_um=None, axislabels=None):
    """Write attributes to a stack."""

    if element_size_um is not None:
        stack.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, l in enumerate(axislabels):
            stack.dims[i].label = l


if __name__ == "__main__":
    main(sys.argv[1:])
