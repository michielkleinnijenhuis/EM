#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.measure import label
from scipy.ndimage.measurements import label as scipy_label
from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects, binary_dilation
from skimage.measure import regionprops

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
    parser.add_argument('--maskDS', default=['_maskDS', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMB', default=['_maskMB', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('-d', '--mode',
                        help='...')
    parser.add_argument('-s', '--min_size', type=int, default=200,
                        help='...')
    parser.add_argument('-S', '--max_size', type=int, default=20000,
                        help='...')
    parser.add_argument('-M', '--mb_intensity', type=float, default=0.25,
                        help='...')
    parser.add_argument('-i', '--slicedim', type=int, default=0,
                        help='...')
    parser.add_argument('-l', '--dolabel', action='store_true',
                        help='...')
    parser.add_argument('-o', '--outpf', default='_labelMA_core',
                        help='...')
    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMB = args.maskMB
    mode = args.mode
    min_size = args.min_size
    max_size = args.max_size
    mb_intensity = args.mb_intensity
    slicedim = args.slicedim
    outpf = args.outpf
    dolabel = args.dolabel
    usempi = args.usempi & ('mpi4py' in sys.modules)

#    elsize = loadh5(datadir, dset_name)[1]

    if mode == '2D':

        fmmname = os.path.join(datadir, dset_name + maskMM[0] + '.h5')
        fdsname = os.path.join(datadir, dset_name + maskDS[0] + '.h5')
        fmbname = os.path.join(datadir, dset_name + maskMB[0] + '.h5')
        fg1name = os.path.join(datadir, dset_name + outpf + '_orig.h5')
        fg2name = os.path.join(datadir, dset_name + outpf + '.h5')

        if usempi:
            # start the mpi communicator
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            fg1 = h5py.File(fg1name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            fg2 = h5py.File(fg2name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            fmm = h5py.File(fmmname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
            fds = h5py.File(fdsname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
            fmb = h5py.File(fmbname, 'r')
        else:
            fg1 = h5py.File(fg1name, 'w')
            fg2 = h5py.File(fg2name, 'w')
            fmm = h5py.File(fmmname, 'r')
            fds = h5py.File(fdsname, 'r')
            fmb = h5py.File(fmbname, 'r')

        n_slices = fmm[maskMM[1]][:,:,:].shape[slicedim]

        if usempi:
            # scatter the slices
            local_nrs = scatter_series(n_slices, comm, size, rank,
                                       MPI.SIGNED_LONG_LONG)[0]
        else:
            local_nrs = np.array(range(0, n_slices), dtype=int)

        outds1 = fg1.create_dataset('stack', fmm[maskMM[1]].shape,
                                    dtype='uint32',
                                    compression="gzip")
        outds2 = fg2.create_dataset('stack', fmm[maskMM[1]].shape,
                                    dtype='uint32',
                                    compression="gzip")

        maxlabel = 0
        fw = np.array([0], dtype='uint32')
        for i in local_nrs:

            MBslc = None
            if slicedim == 0:
                DSslc = fds[maskDS[1]][i,:,:].astype('bool')
                MMslc = fmm[maskMM[1]][i,:,:].astype('bool')
                MBslc = fmb[maskMB[1]][i,:,:].astype('bool')
            elif slicedim == 1:
                DSslc = fds[maskDS[1]][:,i,:].astype('bool')
                MMslc = fmm[maskMM[1]][:,i,:].astype('bool')
                MBslc = fmb[maskMB[1]][:,i,:].astype('bool')
            elif slicedim == 2:
                DSslc = fds[maskDS[1]][:,:,i].astype('bool')
                MMslc = fmm[maskMM[1]][:,:,i].astype('bool')
                MBslc = fmb[maskMB[1]][:,:,i].astype('bool')

            seeds, num = label(np.logical_and(~MMslc, DSslc), return_num=True)
            if usempi:
                seeds += 1000*i  # FIXME: assumed max number of labels in slice is 1000
            else:
                seeds += maxlabel

            if slicedim == 0:
                outds1[i,:,:] = seeds
            elif slicedim == 1:
                outds1[:,i,:] = seeds
            elif slicedim == 2:
                outds1[:,:,i] = seeds

            seeds[MMslc] = 0

            maxlabel += num
            fw.resize(maxlabel + 1)

            rp = regionprops(seeds, intensity_image=MBslc, cache=True)
            mi = {prop.label: (prop.area, prop.mean_intensity)
                  for prop in rp}
            for k, v in mi.items():
                # TODO: other selection criteria?
                if ((v[0] < min_size) or
                    (v[0] > max_size) or
                    v[1] > mb_intensity):
                        seeds[seeds == k] = 0
                        fw[k] = 0
                else:
                        fw[k] = k

            if slicedim == 0:
                outds2[i,:,:] = seeds
            elif slicedim == 1:
                outds2[:,i,:] = seeds
            elif slicedim == 2:
                outds2[:,:,i] = seeds

        filename = os.path.join(datadir, dset_name + outpf + '_fw.npy')
        np.save(filename, fw)

        fmm.close()
        fds.close()
        fg1.close()

    elif mode == '3D':
        maskDS, elsize, al = loadh5(datadir, dset_name + maskDS[0],
                                    fieldname=maskDS[1], dtype='bool')
        maskMM = loadh5(datadir, dset_name + maskMM[0],
                        fieldname=maskMM[1], dtype='bool')[0]

        mask = np.logical_or(binary_dilation(maskMM), ~maskDS)
        remove_small_objects(mask, min_size=100000, in_place=True)

        labels = label(~mask, return_num=False, connectivity=None)
        remove_small_objects(labels, min_size=10000, connectivity=1, in_place=True)

        # remove the unmyelinated axons (largest label)
        rp = regionprops(labels)
        areas = [prop.area for prop in rp]
        labs = [prop.label for prop in rp]
        llab = labs[np.argmax(areas)]
        labels[labels == llab] = 0

        labels = relabel_sequential(labels)[0]

        writeh5(labels, datadir, dset_name + outpf, dtype='int32',
                element_size_um=elsize, axislabels=al)

    else:
        filename = os.path.join(datadir, dset_name + outpf + '.h5')
        f = h5py.File(filename, 'r')

        rp = regionprops(f['stack'][:,:,:], cache=True)
        mi = {prop.label: prop.area for prop in rp}

        labellist = [prop.label for prop in rp]
        maxlabel = np.amax(np.array(labellist))
        fw = np.zeros(maxlabel + 1, dtype='int32')
        for k, v in mi.items():
            if ((mi[k] > min_size) & (mi[k] < max_size)):
                fw[k] = k
        filename = os.path.join(datadir, dset_name + outpf + '_fw.npy')
        np.save(filename, fw)
#        fw = np.load(filename)

        filename = os.path.join(datadir, dset_name + outpf + '_labeled.h5')
        g = h5py.File(filename, 'w')
        outds = g.create_dataset('stack', f['stack'].shape,
                                 dtype='uint32',
                                 compression="gzip")
        outds[:,:,:], num = scipy_label(fw[f['stack'][:,:,:]] != 0)
        # consider relabel sequential, consider removing objects in a single slice, remove small objects
        g.close()
        f.close()


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


def loadh5(datadir, dname, fieldname='stack', dtype=None):
    """"""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]

    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    if 'DIMENSION_LABELS' in f[fieldname].attrs.keys():
        axislabels = [d.label for d in f[fieldname].dims]
    else:
        axislabels = None

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """"""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    if element_size_um is not None:
        g[fieldname].attrs['element_size_um'] = element_size_um
    if axislabels is not None:
        for i, l in enumerate(axislabels):
            g[fieldname].dims[i].label = l

    g.close()


if __name__ == "__main__":
    main(sys.argv[1:])
