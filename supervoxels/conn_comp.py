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
    parser.add_argument('--maskDS', default=['_maskDS', '/stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', '/stack'], nargs=2,
                        help='...')
    parser.add_argument('-d', '--mode',
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
    mode = args.mode
    slicedim = args.slicedim
    outpf = args.outpf
    dolabel = args.dolabel
    usempi = args.usempi & ('mpi4py' in sys.modules)

#    elsize = loadh5(datadir, dset_name)[1]

    if mode == '2D':

        filename = os.path.join(datadir, dset_name + maskMM[0] + '.h5')
        fmm = h5py.File(filename, 'r')
        filename = os.path.join(datadir, dset_name + maskDS[0] + '.h5')
        fds = h5py.File(filename, 'r')

        filename = os.path.join(datadir, dset_name + outpf + '.h5')
        g1 = h5py.File(filename, 'w')
        outds1 = g1.create_dataset('stack', fmm[maskMM[1]].shape,
                                   dtype='uint32',
                                   compression="gzip")
        filename = os.path.join(datadir, dset_name + outpf + '_labeled.h5')
        if dolabel:
            g2 = h5py.File(filename, 'w')
            outds2 = g2.create_dataset('stack', fmm[maskMM[1]].shape,
                                       dtype='uint32',
                                       compression="gzip")

        maxlabel = 0
        for i in range(0, fmm[maskMM[1]][:,:,:].shape[0]):

            if slicedim == 0:
                MMslc = fmm[maskMM[1]][i,:,:].astype('bool')
                DSslc = fds[maskDS[1]][i,:,:].astype('bool')
            elif slicedim == 1:
                MMslc = fmm[maskMM[1]][:,i,:].astype('bool')
                DSslc = fds[maskDS[1]][:,i,:].astype('bool')
            elif slicedim == 2:
                MMslc = fmm[maskMM[1]][:,:,i].astype('bool')
                DSslc = fds[maskDS[1]][:,:,i].astype('bool')

            seeds, num = label(np.logical_and(~MMslc, DSslc), return_num=True)
            seeds += maxlabel
            seeds[MMslc] = 0

            if slicedim == 0:
                outds1[i,:,:] = seeds
            elif slicedim == 1:
                outds1[:,i,:] = seeds
            elif slicedim == 2:
                outds1[:,:,i] = seeds

            maxlabel += num

        rp = regionprops(outds1[:,:,:], cache=True)
        mi = {prop.label: prop.area for prop in rp}

        fw = np.zeros(maxlabel + 1, dtype='int32')
        for k, v in mi.items():
            if ((mi[k] > 200) & (mi[k] < 20000)):
                fw[k] = k
        filename = os.path.join(datadir, dset_name + outpf + '_fw.npy')
        np.save(filename, fw)

        if dolabel:
            # this is more mem intensive, consider moving to seperate function
            outds2[:,:,:] = label(fw[outds1[:,:,:]] != 0)
            g2.close()

        fmm.close()
        fds.close()
        g1.close()

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
            if ((mi[k] > 200) & (mi[k] < 20000)):
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
