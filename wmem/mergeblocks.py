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

from wmem import parse, utils, downsample_blockwise
from wmem import wmeMPI, Image, LabelImage


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
        args.dataslices,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.blockoffset,
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
        images_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        blockoffset=[0, 0, 0],
        fullsize=[],
        is_labelimage=False,
        relabel=False,
        neighbourmerge=False,
        save_fwmap=False,
        blockreduce=[],
        func='np.amax',
        datatype='',
        usempi=False,
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """Merge blocks of data into a single hdf5 file."""

    if blockrange:
        images_in = images_in[blockrange[0]:blockrange[1]]

    mpi = wmeMPI(usempi)

    im = Image(images_in[0], permission='r')
    im.load(mpi.comm, load_data=False)
    props = im.get_props(protective=protective, squeeze=True)
    ndim = im.get_ndim()

    props['dtype'] = datatype or props['dtype']
    props['chunks'] = props['chunks'] or None

    # get the size of the outputfile
    # TODO: option to derive fullsize from dset_names?
    if blockreduce:
        datasize = np.subtract(fullsize, blockoffset)
        outsize = [int(np.ceil(d/np.float(b)))
                   for d, b in zip(datasize, blockreduce)]
        props['elsize'] = [e*b for e, b in zip(im.elsize, blockreduce)]
    else:  # FIXME: 'zyx(c)' stack assumed
        outsize = np.subtract(fullsize, blockoffset)

    if ndim == 4:
        outsize = list(outsize) + [im.ds.shape[3]]  # TODO: flexible insert

    if outputpath.endswith('.ims'):
        mo = LabelImage(outputpath)
        mo.create(comm=mpi.comm)
    else:
        props['shape'] = outsize
        mo = LabelImage(outputpath, **props)
        mo.create(comm=mpi.comm)

    mpi.blocks = [{'path': image_in} for image_in in images_in]
    mpi.nblocks = len(images_in)
    mpi.scatter_series()

    # merge the datasets
    maxlabel = 0
    for i in mpi.series:

        block = mpi.blocks[i]
        try:
            maxlabel = process_block(block['path'], ndim, blockreduce, func,
                                     blockoffset, blocksize, blockmargin,
                                     fullsize,
                                     mo,
                                     is_labelimage, relabel,
                                     neighbourmerge, save_fwmap,
                                     maxlabel, mpi)
            print('processed block {:03d}: {}'.format(i, block['path']))
        except Exception as e:
            print('failed block {:03d}: {}'.format(i, block['path']))
            print(e)

    im.close()
    mo.close()

    return mo


def process_block(image_in, ndim, blockreduce, func,
                  blockoffset, blocksize, margin, fullsize,
                  mo,
                  is_labelimage, relabel, neighbourmerge, save_fwmap,
                  maxlabel, mpi):
    """Write a block of data into a hdf5 file."""

    # open data for reading
    im = Image(image_in, permission='r')
    im.load(mpi.comm, load_data=False)

    # get the indices into the input and output datasets
    # TODO: get indices from attributes
    # TODO: get from mpi.get_blocks
    set_slices_in_and_out(im, mo, blocksize, margin, fullsize)

    # simply copy the data from input to output
    """NOTE:
    it is assumed that the inputs are not 4D labelimages
    """
    if ndim == 4:
        mo.write(im.slice_dataset())
        im.close()
        return
    if ((not is_labelimage) or
            ((not relabel) and
             (not neighbourmerge) and
             (not blockreduce))):
        data = im.slice_dataset()
        mo.write(data)
        im.close()
        return

    # forward map to relabel the blocks in the output
    if relabel:
        # FIXME: make sure to get all data in the block
        fw, maxlabel = relabel_block(im.ds[:], maxlabel, mpi)
        if save_fwmap:
            comps = im.split_path()
            fpath = '{}_{}.npy'.format(comps['base'], comps['int'][1:])
            np.save(fpath, fw)
        if (not neighbourmerge) and (not blockreduce):
            data = im.slice_dataset()
            mo.write(fw[data])
            im.close()
            return
    else:
        ulabels = np.unique(im.ds[:])
        fw = [l for l in range(0, np.amax(ulabels) + 1)]
        fw = np.array(fw)

    # blockwise reduction of input datasets
    if blockreduce is not None:
        data = blockreduce_datablocks(im, mo, blockreduce, func)
        margin = (int(margin[i]/blockreduce[i]) for i in range(0, 3))
        if (not neighbourmerge):
            mo.write(fw[data])
            im.close()
            return
    else:
        data = im.slice_dataset()

    # merge overlapping labels
    fw = merge_overlap(fw, im, mo, data, margin)
    mo.write(fw[data])
    im.close()


def relabel_block(ds_in, maxlabel, mpi=None):
    """Relabel the labelvolume with consecutive labels.

    NOTE:
    relabel from 0, because mpi is unaware of maxlabel before gather
    """
    fw = relabel_sequential(ds_in[:])[1]

    if mpi.enabled:
        # FIXME: only terminates properly when: nblocks % size = 0

        num_labels = np.amax(fw)
        num_labels = mpi.comm.gather(num_labels, root=0)

        if mpi.rank == 0:
            add_labels = [maxlabel + np.sum(num_labels[:i])
                          for i in range(1, mpi.size)]
            add_labels = np.array([maxlabel] + add_labels, dtype='i')
            maxlabel = maxlabel + np.sum(num_labels)
        else:
            add_labels = np.empty(mpi.size)

        add_labels = mpi.comm.bcast(add_labels, root=0)
        fw[1:] += add_labels[mpi.rank]

    else:

        fw[1:] += maxlabel
        if len(fw) > 1:
            maxlabel = np.amax(fw)

    return fw, maxlabel


def blockreduce_datablocks(im, mo, blockreduce, func):
    """Blockwise reduction of the input dataset.

    NOTE:
    upper indices are adapted to select a datablock from the input dataset
    that results in the right shape after blockreduce
    """

    for slc, br in zip(im.slices, blockreduce):
        slc.stop = int(np.ceil(slc.stop / br) * br)

    data = downsample_blockwise.block_reduce(im.slice_dataset(),
                                             block_size=tuple(blockreduce),
                                             func=eval(func))
    if data.ndim == 4:
        data = np.squeeze(data, axis=3)

    for i_slc, o_slc, br, d in zip(im.slices, mo.slices, blockreduce, data.shape):
        # update the indices into the blockreduced input dataset
        i_slc.start = 0
        i_slc.stop = d
        # calculate new indices into the output dataset
        o_slc.start = int(o_slc.start / br)
        o_slc.stop = int(o_slc.stop + d)

    return data


def set_slices_in_and_out(im, mo, blocksize, margin, fullsize, blockoffset=[0, 0, 0]):

    comps = im.split_path()
    _, x, X, y, Y, z, Z = utils.split_filename(comps['file'], blockoffset[:3][::-1])
    (oz, oZ), (iz, iZ) = margins(z, Z, blocksize[0], margin[0], fullsize[0])
    (oy, oY), (iy, iY) = margins(y, Y, blocksize[1], margin[1], fullsize[1])
    (ox, oX), (ix, iX) = margins(x, X, blocksize[2], margin[2], fullsize[2])
    im.slices[0] = slice(iz, iZ, 1)
    im.slices[1] = slice(iy, iY, 1)
    im.slices[2] = slice(ix, iX, 1)
    mo.slices[0] = slice(oz, oZ, 1)
    mo.slices[1] = slice(oy, oY, 1)
    mo.slices[2] = slice(ox, oX, 1)


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


def get_overlap(side, im, mo, data, ixyz, oxyz, margin=[0, 0, 0]):
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
