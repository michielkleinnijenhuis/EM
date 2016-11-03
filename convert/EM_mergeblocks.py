#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np

from skimage.segmentation import relabel_sequential
# from skimage.measure import block_reduce
from skimage.util import view_as_blocks
# from scipy.stats import mode
# from numpy.lib.stride_tricks import as_strided

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):

    parser = ArgumentParser(description="""
        Merge blocks of data into a single .h5 file.""")
    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_names', nargs='*',
                        help='...')
    parser.add_argument('-i', '--inpf', nargs=2, default=['', 'stack'],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2, default=None,
                        help='...')

    parser.add_argument('-b', '--blockoffset', nargs=3, type=int,
                        default=[0, 0, 0],
                        help='zyx')
    parser.add_argument('-p', '--blocksize', nargs=3, type=int,
                        default=[0, 0, 0],
                        help='zyx')
    parser.add_argument('-q', '--margin', nargs=3, type=int,
                        default=[0, 0, 0],
                        help='zyx')
    parser.add_argument('-s', '--fullsize', nargs=3, type=int,
                        default=None,
                        help='zyx')

    parser.add_argument('-l', '--is_labelimage', action='store_true',
                        help='...')
    parser.add_argument('-r', '--relabel', action='store_true',
                        help='...')
    parser.add_argument('-n', '--neighbourmerge', action='store_true',
                        help='...')
    parser.add_argument('-F', '--save_fwmap', action='store_true',
                        help='...')

    parser.add_argument('-B', '--blockreduce', nargs=3, type=int,
                        default=None,
                        help='zyx')
    parser.add_argument('-f', '--func', default='np.amax',
                        help='...')

    parser.add_argument('-m', '--usempi', action='store_true',
                        help='use mpi4py')

    args = parser.parse_args()

    datadir = args.datadir
    dset_names = args.dset_names
    inpf = args.inpf
    outpf = args.outpf

    blockoffset = args.blockoffset
    margin = args.margin
    blocksize = args.blocksize
    fullsize = args.fullsize

    is_labelimage = args.is_labelimage
    relabel = args.relabel
    neighbourmerge = args.neighbourmerge
    save_fwmap = args.save_fwmap
    blockreduce = args.blockreduce
    func = args.func

    usempi = args.usempi & ('mpi4py' in sys.modules)

    fname = dset_names[0] + inpf[0] + '.h5'
    fpath = os.path.join(datadir, fname)

    dset_info = split_filename(fpath, blockoffset)[0]

    if outpf is None:
        outpf = inpf
    gname = dset_info['base'] + outpf[0] + '.h5'
    gpath = os.path.join(datadir, gname)

    if usempi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(comm, rank, size)

        f = h5py.File(fpath, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        g = h5py.File(gpath, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        local_nrs = scatter_series(len(dset_names), comm, size, rank)[0]

    else:
        rank = 0

        f = h5py.File(fpath, 'r')
        g = h5py.File(gpath, 'w')

        local_nrs = np.array(range(0, len(dset_names)), dtype=int)

    fstack = f[inpf[1]]
    elsize, al = get_h5_attributes(fstack)
    ndims = len(fstack.shape)

    # TODO: option to get fullsize from dset_names
    if blockreduce is not None:
        datasize = np.subtract(fullsize, blockoffset)
        outsize = [int(np.ceil(d/b))
                   for d,b in zip(datasize, blockreduce)]
        elsize = [e*b for e, b in zip(elsize, blockreduce)]
    else:  # NOTE: 'zyx(c)' stack assumed
        outsize = np.subtract(fullsize, blockoffset)
        if ndims == 4:
            outsize = outsize + [fstack.shape[3]]

    g.create_dataset(outpf[1], outsize,
                     chunks=fstack.chunks or None,
                     dtype=fstack.dtype,
                     compression=None if usempi else 'gzip')
    gstack = g[outpf[1]]
    write_h5_attributes(gstack, elsize, al)

    f.close()

    maxlabel = 0
    for i in local_nrs:
        dset_name = dset_names[i]
        print('processing block %03d: %s' % (i, dset_name))

        fname = dset_name + inpf[0] + '.h5'
        fpath = os.path.join(datadir, fname)
        f = h5py.File(fpath, 'r')
        fstack = f[inpf[1]]
        dset_info, x, X, y, Y, z, Z = split_filename(fpath, blockoffset)

        (z, Z), (oz, oZ) = margins(z, Z, blocksize[0],
                                   margin[0], fullsize[0])
        (y, Y), (oy, oY) = margins(y, Y, blocksize[1],
                                   margin[1], fullsize[1])
        (x, X), (ox, oX) = margins(x, X, blocksize[2],
                                   margin[2], fullsize[2])

        if ndims == 4:  # NOTE: no 4D labelimages assumed
            gstack[z:Z, y:Y, x:X, :] = fstack[oz:oZ, oy:oY, ox:oX, :]
            f.close()
            continue

        if ((not is_labelimage) or 
            ((not relabel) and (not neighbourmerge) and (not blockreduce))):
            gstack[z:Z, y:Y, x:X] = fstack[oz:oZ, oy:oY, ox:oX]
            f.close()
            continue

        if relabel:
            print('relabeling')

            fw = relabel_sequential(fstack[:, :, :])[1]

            if usempi:
                # FIXME: only terminates properly when: nblocks % size = 0

                num_labels = np.amax(fw)
                print(num_labels)
                num_labels = comm.gather(num_labels, root=0)
                print(num_labels)

                if rank == 0:
                    add_labels = [maxlabel + np.sum(num_labels[:i])
                                  for i in range(1, size)]
                    add_labels = np.array([maxlabel] + add_labels, dtype='i')
                    maxlabel = maxlabel + np.sum(num_labels)
                    print(maxlabel)
                else:
                    add_labels = np.empty(size)

                add_labels = comm.bcast(add_labels, root=0)
                fw[1:] += add_labels[rank]

            else:

                fw[1:] += maxlabel
                maxlabel += np.amax(fw)

        else:

            ulabels = np.unique(fstack[:, :, :])
            fw = [l for l in range(0, np.amax(ulabels) + 1)]
            fw = np.array(fw)

        if neighbourmerge:

            fw = merge_overlap(fw, fstack, gstack,
                               (x, X, y, Y, z, Z),
                               (ox, oX, oy, oY, oz, oZ),
                               margin, fullsize)

        if save_fwmap:

            fname = dset_name + inpf[0] + '.npy'
            fpath = os.path.join(datadir, fname)
            np.save(fpath, fw)


        if blockreduce is not None:

            aZ = int(np.ceil(oZ/blockreduce[0]) * blockreduce[0])
            aY = int(np.ceil(oY/blockreduce[1]) * blockreduce[1])
            aX = int(np.ceil(oX/blockreduce[2]) * blockreduce[2])

            data = block_reduce(fstack[oz:aZ, oy:aY, ox:aX],
                                block_size=tuple(blockreduce),
                                func=eval(func))

            z, y, x = (c / br for c, br in zip( (z, y, x), blockreduce ) )
            idims = data.shape
            Z, Y, X = (c + d for c, d in zip ( (z, y, x), idims) )
            odims = gstack[z:Z, y:Y, x:X].shape
            oZ, oY, oX = (c - 1 if i > o else c
                          for c, i, o in zip( (oZ, oY, oX), idims, odims ) )

        else:

            data = fstack[oz:oZ, oy:oY, ox:oX]

        gstack[z:Z, y:Y, x:X] = fw[data]

        f.close()

    g.close()


# ========================================================================== #
# function defs
# ========================================================================== #


def scatter_series(n, comm, size, rank, SLL=MPI.SIGNED_LONG_LONG):
    """Scatter a series of jobnrs over processes."""

    nrs = np.array(range(0, n), dtype=int)
    local_n = np.ones(size, dtype=int) * n / size
    local_n[0:n % size] += 1
    local_nrs = np.zeros(local_n[rank], dtype=int)
    displacements = tuple(sum(local_n[0:r]) for r in range(0, size))
    comm.Scatterv([nrs, tuple(local_n), displacements, SLL],
                  local_nrs, root=0)

    return local_nrs, tuple(local_n), displacements


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
    x = int(parts[1].split("-")[0]) - blockoffset[2]
    X = int(parts[1].split("-")[1]) - blockoffset[2]
    y = int(parts[2].split("-")[0]) - blockoffset[1]
    Y = int(parts[2].split("-")[1]) - blockoffset[1]
    z = int(parts[3].split("-")[0]) - blockoffset[0]
    Z = int(parts[3].split("-")[1]) - blockoffset[0]

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


def get_overlap(side, fstack, gstack, granges, oranges,
                margin=[0, 0, 0], fullsize=[0, 0, 0]):
    """Return boundary slice of block and its neighbour."""

    x, X, y, Y, z, Z = granges
    ox, oX, oy, oY, oz, oZ = oranges
    # FIXME: need to account for blockoffset
#    ox = max(0, x - margin[0])
#    oX = min(fullsize[0], X + margin[0])
#    oy = max(0, y - margin[1])
#    oY = min(fullsize[1], Y + margin[1])
#    oz = max(0, z - margin[2])
#    oZ = min(fullsize[2], Y + margin[2])

    data_section = None
    nb_section = None

    print(margin, x, X, y, Y, z, Z, ox, oX, oy, oY, oz, oZ)
    if (side == 'xmin') & (x > 0):
#        data_section = fstack[:, :, :margin[2]]
#        if x > 0:
            data_section = fstack[oz:oZ, oy:oY, :margin[2]]
            nb_section = gstack[z:Z, y:Y, x-margin[2]:x]
    elif (side == 'xmax') & (X < gstack.shape[2]):
#        data_section = fstack[:, :, -margin[2]:]
#        if X < gstack.shape[2]:
            data_section = fstack[oz:oZ, oy:oY, -margin[2]:]
            nb_section = gstack[z:Z, y:Y, X:X+margin[2]]
    elif (side == 'ymin') & (y > 0):
#        data_section = fstack[:, :margin[1], :]
#        if y > 0:
            data_section = fstack[oz:oZ, :margin[1], ox:oX]
            nb_section = gstack[z:Z, y-margin[1]:y, x:X]
    elif (side == 'ymax') & (Y < gstack.shape[1]):
#        data_section = fstack[:, -margin[1]:, :]
#        if Y < gstack.shape[1]:
            data_section = fstack[oz:oZ, -margin[1]:, ox:oX]
            nb_section = gstack[oz:oZ, Y:Y+margin[1], ox:oX]
    elif (side == 'zmin') & (z > 0):
#        data_section = fstack[:margin[0], :, :]
#        if z > 0:
            data_section = fstack[:margin[0], oy:oY, ox:oX]
            nb_section = gstack[z-margin[0]:z, y:Y, x:X]
    elif (side == 'zmax') & (Z < gstack.shape[0]):
#        data_section = fstack[-margin[0]:, :, :]
#        if Z < gstack.shape[0]:
            data_section = fstack[-margin[0]:, oy:oY, ox:oX]
            nb_section = gstack[Z:Z+margin[0], y:Y, x:X]

    if nb_section is not None:
        print(side, data_section.shape, nb_section.shape)

    return data_section, nb_section


def merge_overlap(fw, fstack, gstack, granges, oranges,
                  margin=[0, 0, 0], fullsize=[0, 0, 0]):
    """Adapt the forward map to merge neighbouring labels."""

    for side in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:

        data_section, nb_section = get_overlap(side, fstack, gstack,
                                               granges, oranges,
                                               margin, fullsize)
        print(nb_section)
        if nb_section is None:
            continue

        print(nb_section, data_section)
        data_labels = np.trim_zeros(np.unique(data_section))
        print(data_labels)
        for data_label in data_labels:
            print(data_label)

            mask_data = data_section == data_label
            bins = np.bincount(nb_section[mask_data])
            if len(bins) <= 1:
                print("lenbins")
                continue

            nb_label = np.argmax(bins[1:]) + 1
            n_data = np.sum(mask_data)
            n_nb = bins[nb_label]
            if float(n_nb) / float(n_data) < 0.1:
                print("div", n_nb, n_data, nb_label)
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


# adapted version of scikit-image-dev0.13 block_reduce
# it switches to flattened blocks to calculate the (scipy) mode
def block_reduce(image, block_size, func=np.sum, cval=0):
    """Down-sample image by applying function to local blocks.
    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    func : callable
        Function object which is used to calculate the return value for each
        local block. This function must implement an ``axis`` parameter such
        as ``numpy.sum`` or ``numpy.min``.
    cval : float
        Constant padding value if image is not perfectly divisible by the
        block size.
    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
    Examples
    --------
    >>> from skimage.measure import block_reduce
    >>> image = np.arange(3*3*4).reshape(3, 3, 4)
    >>> image # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]],
           [[24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35]]])
    >>> block_reduce(image, block_size=(3, 3, 1), func=np.mean)
    array([[[ 16.,  17.,  18.,  19.]]])
    >>> image_max1 = block_reduce(image, block_size=(1, 3, 4), func=np.max)
    >>> image_max1 # doctest: +NORMALIZE_WHITESPACE
    array([[[11]],
           [[23]],
           [[35]]])
    >>> image_max2 = block_reduce(image, block_size=(3, 1, 4), func=np.max)
    >>> image_max2 # doctest: +NORMALIZE_WHITESPACE
    array([[[27],
            [31],
            [35]]])
    """

    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode='constant',
                   constant_values=cval)

    out = view_as_blocks(image, block_size)

    if func is mode:
        # TODO: implemented restriding here instead of reshape?
        outshape = tuple(image.shape) + tuple([-1])
        out = np.reshape(out, outshape)
        out = mode(out)
    else:
        for i in range(len(out.shape) // 2):
            out = func(out, axis=-1)

    return out


def mode(array, axis=None):
    """Calculate the blockwise mode."""

    smode = np.zeros_like(array)
    for i in range(array.shape[0]):
        print(i)
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                block = array[i,j,k,:].ravel()
                smode[i,j,k] = np.argmax(np.bincount(block))

    return smode


if __name__ == "__main__":
    main(sys.argv)
