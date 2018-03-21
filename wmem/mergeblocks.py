#!/usr/bin/env python

"""Merge blocks of data into a single hdf5 dataset.

example splitting of the full dataset: old
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}.h5 \
$datadir/datastem.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx -p datastem
TODO:
splitting of the full dataset: proposed new
# create group?
# links to subsets of full dataset?
# include coordinates in attributes?
"""

import sys
import argparse
import os

import numpy as np
from skimage.segmentation import relabel_sequential

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")

from wmem import parse, utils, downsample_blockwise


def main(argv):
    """Merge blocks of data into a single hdf5 dataset."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_mergeblocks(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    mergeblocks(
        args.inputfiles,
        args.blockoffset,
        args.blocksize,
        args.margin,
        args.fullsize,
        args.is_labelimage,
        args.relabel,
        args.neighbourmerge,
        args.save_fwmap,
        args.blockreduce,
        args.func,
        args.datatype,
        args.usempi & ('mpi4py' in sys.modules),
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def mergeblocks(
        h5paths_in,
        blockoffset=[0, 0, 0],
        blocksize=[],
        margin=[0, 0, 0],
        fullsize=[],
        is_labelimage=False,
        relabel=False,
        neighbourmerge=False,
        save_fwmap=False,
        blockreduce=[],
        func='np.amax',
        datatype='',
        usempi=False,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Merge blocks of data into a single hdf5 file."""

    # prepare mpi
    mpi_info = utils.get_mpi_info(usempi)
    series = np.array(range(0, len(h5paths_in)), dtype=int)
    if mpi_info['enabled']:
        series = utils.scatter_series(mpi_info, series)[0]

    # TODO: save_steps
    # check output paths
    outpaths = {'out': h5path_out}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5paths_in[0],
                                                    comm=mpi_info['comm'])
    try:
        ndim = ds_in.ndim
    except AttributeError:
        ndim = len(ds_in.dims)

    # get the size of the outputfile
    # TODO: option to derive fullsize from dset_names?
    if blockreduce:
        datasize = np.subtract(fullsize, blockoffset)
        outsize = [int(np.ceil(d/np.float(b)))
                   for d, b in zip(datasize, blockreduce)]
        elsize = [e*b for e, b in zip(elsize, blockreduce)]
    else:  # FIXME: 'zyx(c)' stack assumed
        outsize = np.subtract(fullsize, blockoffset)

    if ndim == 4:
        outsize = list(outsize) + [ds_in.shape[3]]  # TODO: flexible insert

    datatype = datatype or ds_in.dtype
    chunks = ds_in.chunks or None

    h5file_in.close()

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, outsize, datatype,
                                        h5path_out,
                                        chunks=chunks,
                                        element_size_um=elsize,
                                        axislabels=axlab,
                                        usempi=usempi,
                                        comm=mpi_info['comm'],
                                        rank=mpi_info['rank'])

    # merge the datasets
    maxlabel = 0
    for i in series:
        h5path_in = h5paths_in[i]
        try:
            maxlabel = process_block(h5path_in, ndim, blockreduce, func,
                                     blockoffset, blocksize, margin, fullsize,
                                     ds_out,
                                     is_labelimage, relabel,
                                     neighbourmerge, save_fwmap,
                                     maxlabel, usempi, mpi_info)
            print('processed block {:03d}: {}'.format(i, h5path_in))
        except:
            print('failed block {:03d}: {}'.format(i, h5path_in))

    # close and return
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def process_block(h5path_in, ndim, blockreduce, func,
                  blockoffset, blocksize, margin, fullsize,
                  ds_out,
                  is_labelimage, relabel, neighbourmerge, save_fwmap,
                  maxlabel, usempi, mpi_info):
    """Write a block of data into a hdf5 file."""

    # open data for reading
    h5file_in, ds_in, _, _ = utils.h5_load(h5path_in)

    # get the indices into the input and output datasets
    # TODO: get indices from attributes
    """NOTE:
    # x, X, y, Y, z, Z are indices into the full dataset
    # ix, iX, iy, iY, iz, iZ are indices into the input dataset
    # ox, oX, oy, oY, oz, oZ are indices into the output dataset
    """
    _, x, X, y, Y, z, Z = utils.split_filename(h5file_in.filename,
                                               [blockoffset[2],
                                                blockoffset[1],
                                                blockoffset[0]])
    (oz, oZ), (iz, iZ) = margins(z, Z, blocksize[0],
                                 margin[0], fullsize[0])
    (oy, oY), (iy, iY) = margins(y, Y, blocksize[1],
                                 margin[1], fullsize[1])
    (ox, oX), (ix, iX) = margins(x, X, blocksize[2],
                                 margin[2], fullsize[2])
    ixyz = ix, iX, iy, iY, iz, iZ
    oxyz = ox, oX, oy, oY, oz, oZ

    # simply copy the data from input to output
    """NOTE:
    it is assumed that the inputs are not 4D labelimages
    """
    if ndim == 4:
        ds_out[oz:oZ, oy:oY, ox:oX, :] = ds_in[iz:iZ, iy:iY, ix:iX, :]
        h5file_in.close()
        return
    if ((not is_labelimage) or
            ((not relabel) and
             (not neighbourmerge) and
             (not blockreduce))):
        ds_out[oz:oZ, oy:oY, ox:oX] = ds_in[iz:iZ, iy:iY, ix:iX]
        h5file_in.close()
        return

    # forward map to relabel the blocks in the output
    if relabel:
        fw, maxlabel = relabel_block(ds_in, maxlabel, mpi_info)
        if save_fwmap:
            root = os.path.splitext(h5file_in.filename)[0]
            fpath = '{}_{}.npy'.format(root, ds_in.name[1:])
            np.save(fpath, fw)
        if (not neighbourmerge) and (not blockreduce):
            ds_out[oz:oZ, oy:oY, ox:oX] = fw[ds_in[iz:iZ, iy:iY, ix:iX]]
            h5file_in.close()
            return
    else:
        ulabels = np.unique(ds_in[:])
        fw = [l for l in range(0, np.amax(ulabels) + 1)]
        fw = np.array(fw)

    # blockwise reduction of input datasets
    if blockreduce is not None:
        data, ixyz, oxyz = blockreduce_datablocks(ds_in, ds_out,
                                                  ixyz, oxyz,
                                                  blockreduce, func)
        ix, iX, iy, iY, iz, iZ = ixyz
        ox, oX, oy, oY, oz, oZ = oxyz
        margin = (int(margin[i]/blockreduce[i]) for i in range(0, 3))
        if (not neighbourmerge):
            ds_out[oz:oZ, oy:oY, ox:oX] = fw[data]
            h5file_in.close()
            return
    else:
        data = ds_in[iz:iZ, iy:iY, ix:iX]

    # merge overlapping labels
    fw = merge_overlap(fw, data, ds_out, oxyz, ixyz, margin)
    ds_out[oz:oZ, oy:oY, ox:oX] = fw[data]
    h5file_in.close()


def relabel_block(ds_in, maxlabel, mpi_info=None):
    """Relabel the labelvolume with consecutive labels."""

    """NOTE:
    relabel from 0, because mpi is unaware of maxlabel before gather
    """
    fw = relabel_sequential(ds_in[:])[1]

    if mpi_info['enabled']:
        # FIXME: only terminates properly when: nblocks % size = 0
        comm = mpi_info['comm']
        rank = mpi_info['rank']
        size = mpi_info['size']

        num_labels = np.amax(fw)
        num_labels = comm.gather(num_labels, root=0)

        if rank == 0:
            add_labels = [maxlabel + np.sum(num_labels[:i])
                          for i in range(1, size)]
            add_labels = np.array([maxlabel] + add_labels, dtype='i')
            maxlabel = maxlabel + np.sum(num_labels)
        else:
            add_labels = np.empty(size)

        add_labels = comm.bcast(add_labels, root=0)
        fw[1:] += add_labels[rank]

    else:

        fw[1:] += maxlabel
        if len(fw) > 1:
            maxlabel = np.amax(fw)

    return fw, maxlabel


def blockreduce_datablocks(ds_in, ixyz, oxyz, blockreduce, func):
    """Blockwise reduction of the input dataset."""

    ox, oX, oy, oY, oz, oZ = oxyz
    ix, iX, iy, iY, iz, iZ = ixyz

    # adapt the upper indices
    """NOTE:
    upper indices are adapted to select a datablock from the input dataset
    that results in the right shape after blockreduce
    """
    aZ = int(np.ceil(iZ / blockreduce[0]) * blockreduce[0])
    aY = int(np.ceil(iY / blockreduce[1]) * blockreduce[1])
    aX = int(np.ceil(iX / blockreduce[2]) * blockreduce[2])

    data = downsample_blockwise.block_reduce(ds_in[iz:aZ, iy:aY, ix:aX],
                                             block_size=tuple(blockreduce),
                                             func=eval(func))
    if data.ndim == 4:
        data = np.squeeze(data, axis=3)

    # update the indices into the blockreduced input dataset
    iz, iy, ix = (0, 0, 0)
    iZ, iY, iX = tuple(data.shape)
    # calculate new indices into the output dataset
    oz, oy, ox = (int(c / br) for c, br in zip((oz, oy, ox), blockreduce))
    oZ, oY, oX = (int(c + d) for c, d in zip((oZ, oY, oX), data.shape))

    return data, (ix, iX, iy, iY, iz, iZ), (ox, oX, oy, oY, oz, oZ)


def margins(fc, fC, blocksize, margin, fullsize):
    """Return lower coordinate (fullstack and block) corrected for margin."""

    if fc == 0:
        bc = 0
    else:
        bc = 0 + margin
        fc += margin

    if fC == fullsize:
        bC = bc + blocksize  # FIXME
    else:
        bC = bc + blocksize
        fC -= margin

    return (fc, fC), (bc, bC)


def get_overlap(side, ds_in, ds_out, ixyz, oxyz, margin=[0, 0, 0]):
    """Return boundary slice of block and its neighbour."""

    ix, iX, iy, iY, iz, iZ = ixyz
    ox, oX, oy, oY, oz, oZ = oxyz
    # FIXME: need to account for blockoffset

    data_section = None
    nb_section = None

    if (side == 'xmin') & (ox > 0):
        data_section = ds_in[iz:iZ, iy:iY, :margin[2]]
        nb_section = ds_out[oz:oZ, oy:oY, ox-margin[2]:ox]
    elif (side == 'xmax') & (oX < ds_out.shape[2]):
        data_section = ds_in[iz:iZ, iy:iY, -margin[2]:]
        nb_section = ds_out[oz:oZ, oy:oY, oX:oX+margin[2]]
    elif (side == 'ymin') & (oy > 0):
        data_section = ds_in[iz:iZ, :margin[1], ix:iX]
        nb_section = ds_out[oz:oZ, oy-margin[1]:oy, ox:oX]
    elif (side == 'ymax') & (oY < ds_out.shape[1]):
        data_section = ds_in[iz:iZ, -margin[1]:, ix:iX]
        nb_section = ds_out[oz:oZ, oY:oY+margin[1], ox:oX]
    elif (side == 'zmin') & (oz > 0):
        data_section = ds_in[:margin[0], iy:iY, ix:iX]
        nb_section = ds_out[oz-margin[0]:oz, oy:oY, ox:oX]
    elif (side == 'zmax') & (oZ < ds_out.shape[0]):
        data_section = ds_in[-margin[0]:, iy:iY, ix:iX]
        nb_section = ds_out[oZ:oZ+margin[0], oy:oY, ox:oX]

    return data_section, nb_section


def merge_overlap(fw, ds_in, ds_out, ixyz, oxyz, margin=[0, 0, 0]):
    """Adapt the forward map to merge neighbouring labels."""

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        ds, ns = get_overlap(side, ds_in, ds_out, ixyz, oxyz, margin)

        if ns is None:
            continue

        data_labels = np.trim_zeros(np.unique(ds))
        for data_label in data_labels:

            mask_data = ds == data_label
            bins = np.bincount(ns[mask_data])
            if len(bins) <= 1:
                continue

            nb_label = np.argmax(bins[1:]) + 1
            n_data = np.sum(mask_data)
            n_nb = bins[nb_label]
            if float(n_nb) / float(n_data) < 0.1:
                continue

            fw[data_label] = nb_label

    return fw


def get_sections(side, ds_in, ds_out, xyz):
    """Return boundary slice of block and its neighbour."""

    x, X, y, Y, z, Z = xyz
    nb_section = None

    if side == 'xmin':
        data_section = ds_in[:, :, 0]
        if x > 0:
            nb_section = ds_out[z:Z, y:Y, x-1]
    elif side == 'xmax':
        data_section = ds_in[:, :, -1]
        if X < ds_out.shape[2]:
            nb_section = ds_out[z:Z, y:Y, X]
    elif side == 'ymin':
        data_section = ds_in[:, 0, :]
        if y > 0:
            nb_section = ds_out[z:Z, y-1, x:X]
    elif side == 'ymax':
        data_section = ds_in[:, -1, :]
        if Y < ds_out.shape[1]:
            nb_section = ds_out[z:Z, Y, x:X]
    elif side == 'zmin':
        data_section = ds_in[0, :, :]
        if z > 0:
            nb_section = ds_out[z-1, y:Y, x:X]
    elif side == 'zmax':
        data_section = ds_in[-1, :, :]
        if Z < ds_out.shape[0]:
            nb_section = ds_out[Z, y:Y, x:X]

    return data_section, nb_section


def merge_neighbours(fw, ds_in, ds_out, xyz):
    """Adapt the forward map to merge neighbouring labels."""

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        data_section, nb_section = get_sections(side, ds_in, ds_out, xyz)
        if nb_section is None:
            continue

        data_labels = np.trim_zeros(np.unique(data_section))
        for data_label in data_labels:

            mask_data = data_section == data_label
            bins = np.bincount(nb_section[mask_data])
            if len(bins) <= 1:
                continue

            nb_label = np.argmax(bins[1:]) + 1
            n_data = np.sum(mask_data)
            n_nb = bins[nb_label]
            if float(n_nb) / float(n_data) < 0.1:
                continue

            fw[data_label] = nb_label
            print('%s: mapped label %d to %d' % (side, data_label, nb_label))

    return fw


if __name__ == "__main__":
    main(sys.argv)
