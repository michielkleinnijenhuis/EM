#!/usr/bin/env python

"""
python EM_mergeblocks.py ...
"""

import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):
    """Merge blocks of data into a single .h5 file."""

    parser = ArgumentParser(description="""
        Merge blocks of data into a single .h5 file.""")
    parser.add_argument('outputfile',
                        help='...')
    parser.add_argument('-i', '--inputfiles', nargs='*',
                        help='...')
    parser.add_argument('-f', '--field', default='stack',
                        help='...')
    parser.add_argument('-b', '--blockoffset', nargs='*', type=int,
                        help='...')
    parser.add_argument('-m', '--usempi', action='store_true',
                        help='use mpi4py')
    args = parser.parse_args()

    outputfile = args.outputfile
    inputfiles = args.inputfiles
    field = args.field
    blockoffset = args.blockoffset
    usempi = args.usempi & ('mpi4py' in sys.modules)

    ranges = np.empty([len(inputfiles), 6], dtype='int')
    for i, filename in enumerate(inputfiles):
        a = find_ranges_from_filename(filename, blockoffset)
        ranges[i, :] = a

    maxX = np.amax(ranges[:, 1])
    maxY = np.amax(ranges[:, 3])
    maxZ = np.amax(ranges[:, 5])

    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # scatter the pairs
        local_nrs, n, disp = scatter_series(len(inputfiles), comm, size, rank,
                                            MPI.SIGNED_LONG_LONG)
    else:
        local_nrs = np.array(range(0, len(inputfiles)), dtype=int)
        rank = 0

    f = h5py.File(inputfiles[0], 'r')
    ndims = len(f[field].shape)

    if ndims == 3:
        dshape = (maxZ, maxY, maxX)
    elif ndims == 4:
        dshape = (maxZ, maxY, maxX, f[field].shape[3])

    if rank == 0:

        datatype = f[field].dtype

        g = h5py.File(outputfile, 'w')

        if f[field].chunks is not None:
            # NOTE: cannot combine chunks with zip filters
            outds = g.create_dataset(field, dshape,
                                     chunks=tuple(f[field].chunks),
                                     dtype=datatype)
        else:
            outds = g.create_dataset(field, dshape,
                                     dtype=datatype,
                                     compression='gzip')
        try:
            outds.attrs['element_size_um'] = \
                f[field].attrs['element_size_um']
        except:
            pass
        try:
            for i, d in enumerate(f[field].dims):
                outds.dims[i].label = d.label
        except:
            pass

        g.close()

    f.close()

    if usempi:
        comm.Barrier()
        g = h5py.File(outputfile, 'a', driver='mpio', comm=MPI.COMM_WORLD)
    else:
        g = h5py.File(outputfile, 'a')


    for i in local_nrs:

        f = h5py.File(inputfiles[i], 'r')

        x, X, y, Y, z, Z = tuple(ranges[i, :])

        if ndims == 3:
            g[field][z:Z, y:Y, x:X] = f[field][:, :, :]
        elif ndims == 4:
            g[field][z:Z, y:Y, x:X, :] = f[field][:, :, :, :]

        f.close()

    g.close()


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


def find_ranges_from_filename(filename, blockoffset):
    """Chop up filename to find x,y,z ranges."""

    head, tail = os.path.split(filename)
    fname, ext = os.path.splitext(tail)
    parts = fname.split("_")
    x = int(parts[1].split("-")[0]) - blockoffset[0]
    X = int(parts[1].split("-")[1]) - blockoffset[0]
    y = int(parts[2].split("-")[0]) - blockoffset[1]
    Y = int(parts[2].split("-")[1]) - blockoffset[1]
    z = int(parts[3].split("-")[0]) - blockoffset[2]
    Z = int(parts[3].split("-")[1]) - blockoffset[2]

    return np.array([x, X, y, Y, z, Z])


if __name__ == "__main__":
    main(sys.argv)
