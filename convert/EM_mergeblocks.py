#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np
from skimage.segmentation import relabel_sequential


def main(argv):

    parser = ArgumentParser(description="""
        Merge blocks of data into a single .h5 file.""")
    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('outputfile',
                        help='...')
    parser.add_argument('-i', '--inputfiles', nargs='*',
                        help='...')
    parser.add_argument('-t', '--postfix', default='',
                        help='...')
    parser.add_argument('-f', '--field', default='stack',
                        help='...')
    parser.add_argument('-m', '--mask', nargs=2,
                        help='...')
    parser.add_argument('-l', '--outlayout',
                        help='...')
    parser.add_argument('-b', '--blockoffset', nargs=3, type=int, default=[0, 0, 0],
                        help='...')
    parser.add_argument('-p', '--blocksize', nargs=3, type=int, default=[0, 0, 0],
                        help='...')
    parser.add_argument('-q', '--margin', nargs=3, type=int, default=[0, 0, 0],
                        help='...')
    parser.add_argument('-s', '--fullsize', nargs=3, type=int, default=None,
                        help='...')
    parser.add_argument('-c', '--chunksize', nargs='*', type=int,
                        help='...')
    parser.add_argument('-e', '--element_size_um', nargs='*', type=float,
                        help='...')
    parser.add_argument('-r', '--relabel', action='store_true',
                        help='...')
    parser.add_argument('-n', '--neighbourmerge', action='store_true',
                        help='...')
    args = parser.parse_args()

    datadir = args.datadir
    outputfile = args.outputfile
    inputfiles = args.inputfiles
    postfix = args.postfix
    field = args.field
    mask = args.mask
    chunksize = args.chunksize
    blockoffset = args.blockoffset
    margin = args.margin
    blocksize = args.blocksize
    fullsize = args.fullsize
    element_size_um = args.element_size_um
    outlayout = args.outlayout
    relabel = args.relabel
    neighbourmerge = args.neighbourmerge

    firstfile = os.path.join(datadir, inputfiles[0] + postfix + '.h5')
    f = h5py.File(firstfile, 'r')
    ndims = len(f[field].shape)
    g = create_dset(outputfile, f, field, ndims,
                    chunksize, element_size_um, outlayout)
    f.close()

    maxlabel = 0
    for inputfile in inputfiles:

        filepath = os.path.join(datadir, inputfile + postfix + '.h5')
        f = h5py.File(filepath, 'r')
        dset_info, x, X, y, Y, z, Z = split_filename(filepath, blockoffset)
#         if mask is not None:
#             dset_info['postfix'] = mask[0]
#             maskfilename = dataset_name(dset_info)
#             m = h5py.File(os.path.join(dset_info['datadir'],
#                                        maskfilename + '.h5'), 'r')
#             ma = np.array(m[mask[0]][:,:,:], dtype='bool')

        (x, X), (ox, oX) = margins(x, X, blocksize[0], margin[0], fullsize[0])
        (y, Y), (oy, oY) = margins(y, Y, blocksize[1], margin[1], fullsize[1])
        (z, Z), (oz, oZ) = margins(z, Z, blocksize[2], margin[2], fullsize[2])
#         print(x, X, y, Y, z, Z)
#         print(ox, oX, oy, oY, oz, oZ)

        for i, newmax in enumerate([Z, Y, X]):
            if newmax > g[field].shape[i]:
                g[field].resize(newmax, i)
#         print(f[field].shape)
#         print(g[field].shape)

        if ndims == 4:
            g[field][z:Z, y:Y, x:X, :] = f[field][oz:oZ, oy:oY, ox:oX, :]
            continue

        if (not relabel) and (not neighbourmerge):
            g[field][z:Z, y:Y, x:X] = f[field][oz:oZ, oy:oY, ox:oX]
            continue

        if relabel:
            print('relabeling')
            fw = relabel_sequential(f[field][:, :, :], maxlabel + 1)[1]
            maxlabel = np.amax(fw)
        else:
            labels = np.unique(f[field][:, :, :])
            fw = np.zeros(np.amax(labels))
            for l in labels:
                fw[l] = l
        if neighbourmerge:
            print('merging neighbours')
            if fullsize is None:
#                 fw = merge_neighbours(fw, f[field], g[field],
#                                       (x, X, y, Y, z, Z))
                fw = merge_overlap(dset_info, fw, f[field], g[field],
                                   (x, X, y, Y, z, Z))
            else:
                pass

        g[field][z:Z, y:Y, x:X] = fw[f[field][oz:oZ, oy:oY, ox:oX]]

        f.close()

    g.close()


# ========================================================================== #
# function defs
# ========================================================================== #


def dataset_name(dset_info):
    """Return the basename of the dataset."""

    nf = dset_info['nzfills']
    dname = dset_info['base'] + \
                '_' + str(dset_info['x']).zfill(nf) + \
                '-' + str(dset_info['X']).zfill(nf) + \
                '_' + str(dset_info['y']).zfill(nf) + \
                '-' + str(dset_info['Y']).zfill(nf) + \
                '_' + str(dset_info['z']).zfill(nf) + \
                '-' + str(dset_info['Z']).zfill(nf) + \
                dset_info['postfix']

    return dname


def split_filename(filename, blockoffset=[0, 0, 0]):
    """Extract the data indices from the filename."""

    datadir, tail = os.path.split(filename)
    fname = os.path.splitext(tail)[0]
    parts = fname.split("_")
    x = int(parts[1].split("-")[0]) - blockoffset[0]
    X = int(parts[1].split("-")[1]) - blockoffset[0]
    y = int(parts[2].split("-")[0]) - blockoffset[1]
    Y = int(parts[2].split("-")[1]) - blockoffset[1]
    z = int(parts[3].split("-")[0]) - blockoffset[2]
    Z = int(parts[3].split("-")[1]) - blockoffset[2]

    dset_info = {'datadir': datadir, 'base': parts[0],
                 'nzfills': len(parts[1].split("-")[0]),
                 'postfix': '_'.join(parts[4:]),
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}

    return dset_info, x, X, y, Y, z, Z


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


def create_dset(outputfile, f, field, ndims,
                chunksize, element_size_um, outlayout):
    """Prepare the dataset to hold the merged volume."""

    if not chunksize:
        try:
            chunksize = f[field].chunks
        except:
            pass

    maxshape = [None] * ndims

    g = h5py.File(outputfile, 'w')
    outds = g.create_dataset(field, np.zeros(len(f[field].shape)),
                             chunks=chunksize,
                             dtype=f[field].dtype,
                             maxshape=maxshape,
                             compression="gzip")

    if element_size_um:
        outds.attrs['element_size_um'] = element_size_um
    else:
        try:
            outds.attrs['element_size_um'] = f[field].attrs['element_size_um']
        except:
            pass

    if outlayout:
        for i, l in enumerate(outlayout):
            outds.dims[i].label = l
    else:
        try:
            for i, d in enumerate(f[field].dims):
                outds.dims[i].label = d.label
        except:
            pass

    return g


def get_overlap(side, fstack, gstack, granges,
                margin=[0, 0, 0], fullsize=[0, 0, 0]):
    """Return boundary slice of block and its neighbour."""

    x, X, y, Y, z, Z = granges
    nb_section = None

    if side == 'xmin':
        data_section = fstack[:, :, 0:margin[0]]
        if x > 0:
            nb_section = gstack[z:Z, y:Y, x-margin[0]:]
    elif side == 'xmax':
        data_section = fstack[:, :, :-margin[0]]
        if X < fullsize[0]:
            nb_section = gstack[z:Z, y:Y, X:X+margin[0]]
    elif side == 'ymin':
        data_section = fstack[:, 0:margin[1], :]
        if y > 0:
            nb_section = gstack[z:Z, y-margin[1]:, x:X]
    elif side == 'ymax':
        data_section = fstack[:, :-margin[1], :]
        if Y < fullsize[1]:
            nb_section = gstack[z:Z, Y:Y+margin[1], x:X]
    elif side == 'zmin':
        data_section = fstack[0:margin[2], :, :]
        if z > 0:
            nb_section = gstack[z-margin[2]:, y:Y, x:X]
    elif side == 'zmax':
        data_section = fstack[:-margin[2], :, :]
        if Z < fullsize[2]:
            nb_section = gstack[Z:Z+margin[2], y:Y, x:X]

    return data_section, nb_section


def merge_overlap(fw, fstack, gstack, granges):
    """Adapt the forward map to merge neighbouring labels."""

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        data_section, nb_section = get_overlap(side, fstack, gstack, granges)
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


def get_sections(side, fstack, gstack, granges):
    """Return boundary slice of block and its neighbour."""

    x, X, y, Y, z, Z = granges
    nb_section = None

    if side == 'xmin':
        data_section = fstack[:, :, 0]
        if x > 0:
            nb_section = gstack[z:Z, y:Y, x-1]
    elif side == 'xmax':
        data_section = fstack[:, :, -1]
        if X < gstack.shape[2]:
            nb_section = gstack[z:Z, y:Y, X]
    elif side == 'ymin':
        data_section = fstack[:, 0, :]
        if y > 0:
            nb_section = gstack[z:Z, y-1, x:X]
    elif side == 'ymax':
        data_section = fstack[:, -1, :]
        if Y < gstack.shape[1]:
            nb_section = gstack[z:Z, Y, x:X]
    elif side == 'zmin':
        data_section = fstack[0, :, :]
        if z > 0:
            nb_section = gstack[z-1, y:Y, x:X]
    elif side == 'zmax':
        data_section = fstack[-1, :, :]
        if Z < gstack.shape[0]:
            nb_section = gstack[Z, y:Y, x:X]

    return data_section, nb_section


def merge_neighbours(fw, fstack, gstack, granges):
    """Adapt the forward map to merge neighbouring labels."""

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        data_section, nb_section = get_sections(side, fstack, gstack, granges)
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
