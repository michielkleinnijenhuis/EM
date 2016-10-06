#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


def main(argv):
    """..."""

    # parse arguments
    parser = ArgumentParser(description=""".""")
    parser.add_argument('datadir', help='...')
    parser.add_argument('dset_name', help='...')
    parser.add_argument('-p', '--input_probs', nargs=2, 
                        default=['_myelin', '_membrane'], help='...')
    parser.add_argument('-f', '--fieldnamesin', nargs=2, 
                        default=['stack', 'stack'], help='...')
    parser.add_argument('-w', '--input_watershed', default='_ws', help='...')
    parser.add_argument('-o', '--output_postfix', default='_per', help='...')
    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')
    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    input_probs = args.input_probs
    fieldnamesin = args.fieldnamesin
    input_watershed = args.input_watershed
    output_postfix = args.output_postfix
    usempi = args.usempi & ('mpi4py' in sys.modules)


    probmask1, elsize = loadh5(datadir, dset_name + input_probs[0], 
                               fieldname=fieldnamesin[0])
    probmask2, elsize = loadh5(datadir, dset_name + input_probs[1], 
                               fieldname=fieldnamesin[1])
    ws = loadh5(datadir, dset_name + input_watershed)[0]
    labels = np.unique(ws)[1:]  # assumes there is a 0 background label!

    if usempi:
        # start the mpi communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        datatype = MPI.DOUBLE.Create_contiguous(3).Commit()
        # scatter the pairs
        local_nrs, n, disp = scatter_series(len(labels), comm, size, rank,
                                            MPI.SIGNED_LONG_LONG)
        if rank == 0:
            percGathered = np.zeros((len(labels), 3), dtype='float')
        else:
            percGathered = None
    else:
        local_nrs = np.array(range(0, len(labels)), dtype=int)


    perc = []
    for i in local_nrs:
        mask = ws==labels[i]
        maskbound = binary_dilation(mask) - mask
        nvox = maskbound.sum()
        nvox_myel = np.logical_and(maskbound, probmask1).sum()
        nvox_unmyel = np.logical_and(maskbound, probmask2).sum()
        p1 = float(nvox_myel)/nvox
        p2 = float(nvox_unmyel)/nvox
        perc.append([p1, p2, p1-p2])
    perc = np.array(perc, dtype='float')

    if usempi:
        comm.Barrier()
        comm.Gatherv(perc, [percGathered, n, disp, datatype])
        datatype.Free()
    else:
        percGathered = perc

    if ((not usempi) or rank == 0):
        per = np.zeros_like(ws, dtype='float')
        for i, l in enumerate(labels):
            per[ws==l] = percGathered[i,2]
        writeh5(per, datadir, dset_name + output_postfix, 
                dtype='float', element_size_um=elsize)
        fname = os.path.join(datadir, dset_name + output_postfix + '.txt')
        out = np.concatenate((labels[:,None], percGathered), axis=1)
        np.savetxt(fname, out, fmt= ['%6d', '%6.4f', '%6.4f', '%6.4f'])


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


def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')
    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:,:]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:,:,:]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:,:,:,:]
    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    f.close()
    return stack, element_size_um


def writeh5(stack, datadir, fp_out, fieldname='stack', dtype='uint16', element_size_um=None):
    """"""
    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")
    if len(stack.shape) == 2:
        g[fieldname][:,:] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:,:,:] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:,:,:,:] = stack
    if element_size_um is not None:
        g[fieldname].attrs['element_size_um'] = element_size_um
    g.close()


if __name__ == "__main__":
    main(sys.argv)
