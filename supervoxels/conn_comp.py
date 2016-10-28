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
    parser.add_argument('-i', '--inpf', default='_labelMA_2Dcore',
                        help='...')
    parser.add_argument('-o', '--outpf', default='_labelMA_2Dcore_fw_',
                        help='...')
    parser.add_argument('--maskDS', default=['_maskDS', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMB', default=['_maskMB', 'stack'], nargs=2,
                        help='...')
    parser.add_argument('-d', '--slicedim', type=int, default=0,
                        help='...')

    parser.add_argument('-M', '--mode',
                        help='...')
    parser.add_argument('-p', '--map_propnames', nargs='*',
                        help='...')

    parser.add_argument('-a', '--min_area', type=int, default=None,
                        help='...')
    parser.add_argument('-A', '--max_area', type=int, default=None,
                        help='...')
    parser.add_argument('-I', '--max_intensity_mb', type=float, default=None,
                        help='...')
    parser.add_argument('-E', '--max_eccentricity', type=float, default=None,
                        help='...')
    parser.add_argument('-e', '--min_euler_number', type=float, default=None,
                        help='...')
    parser.add_argument('-s', '--min_solidity', type=float, default=None,
                        help='...')
    parser.add_argument('-x', '--min_extent', type=float, default=None,
                        help='...')

    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    inpf = args.inpf
    outpf = args.outpf
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMB = args.maskMB
    mode = args.mode
    slicedim = args.slicedim
    map_propnames = args.map_propnames
    min_area = args.min_area
    max_area = args.max_area
    max_intensity_mb = args.max_intensity_mb
    max_eccentricity = args.max_eccentricity
    min_euler_number = args.min_euler_number
    min_solidity = args.min_solidity
    min_extent = args.min_extent
    usempi = args.usempi & ('mpi4py' in sys.modules)

    if mode == '3D':

        maskDS, elsize, al = loadh5(datadir, dset_name + maskDS[0],
                                    fieldname=maskDS[1], dtype='bool')
        maskMM = loadh5(datadir, dset_name + maskMM[0],
                        fieldname=maskMM[1], dtype='bool')[0]

        mask = np.logical_or(binary_dilation(maskMM), ~maskDS)
        remove_small_objects(mask, min_size=100000, in_place=True)

        labels = label(~mask, return_num=False, connectivity=None)
        remove_small_objects(labels, min_size=min_area,
                             connectivity=1, in_place=True)

        # remove the unmyelinated axons (largest label)
        rp = regionprops(labels)
        areas = [prop.area for prop in rp]
        labs = [prop.label for prop in rp]
        llab = labs[np.argmax(areas)]
        labels[labels == llab] = 0

        labels = relabel_sequential(labels)[0]

        writeh5(labels, datadir, dset_name + outpf, dtype='int32',
                element_size_um=elsize, axislabels=al)

    elif mode == '2D':

        fg1name = os.path.join(datadir, dset_name + outpf + '.h5')
        fdsname = os.path.join(datadir, dset_name + maskDS[0] + '.h5')
        fmmname = os.path.join(datadir, dset_name + maskMM[0] + '.h5')

        if usempi:
            # start the mpi communicator
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            fg1 = h5py.File(fg1name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            fds = h5py.File(fdsname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
            fmm = h5py.File(fmmname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        else:
            fg1 = h5py.File(fg1name, 'w')
            fds = h5py.File(fdsname, 'r')
            fmm = h5py.File(fmmname, 'r')

        n_slices = fmm[maskMM[1]].shape[slicedim]

        if usempi:
            # scatter the slices
            local_nrs = scatter_series(n_slices, comm, size, rank,
                                       MPI.SIGNED_LONG_LONG)[0]
        else:
            local_nrs = np.array(range(0, n_slices), dtype=int)

        outds1 = fg1.create_dataset('stack', fmm[maskMM[1]].shape,
                                    dtype='uint32',
                                    compression="gzip")

        maxlabel = 0
        for i in local_nrs:
            print("processing slice %d" % i)

            MBslc = None
            if slicedim == 0:
                DSslc = fds[maskDS[1]][i,:,:].astype('bool')
                MMslc = fmm[maskMM[1]][i,:,:].astype('bool')
            elif slicedim == 1:
                DSslc = fds[maskDS[1]][:,i,:].astype('bool')
                MMslc = fmm[maskMM[1]][:,i,:].astype('bool')
            elif slicedim == 2:
                DSslc = fds[maskDS[1]][:,:,i].astype('bool')
                MMslc = fmm[maskMM[1]][:,:,i].astype('bool')

            labels, num = label(np.logical_and(~MMslc, DSslc), return_num=True)
            if usempi:
                # FIXME: assumed max number of labels in slice is 1000
                labels[~MMslc] += 1000 * i
            else:
                labels[~MMslc] += maxlabel

            if slicedim == 0:
                outds1[i,:,:] = labels
            elif slicedim == 1:
                outds1[:,i,:] = labels
            elif slicedim == 2:
                outds1[:,:,i] = labels

            maxlabel += num

        filename = os.path.join(datadir, dset_name + outpf + '.npy')
        np.save(filename, np.array([maxlabel]))

        fg1.close()
        fmm.close()
        fds.close()

    elif mode == "2Dfilter":

        out = dset_name + outpf + '_'

        filename = os.path.join(datadir, dset_name + inpf + '.h5')
        f = h5py.File(filename, 'r')
        fmbname = os.path.join(datadir, dset_name + maskMB[0] + '.h5')
        fmb = h5py.File(fmbname, 'r')

        try:
            filename = os.path.join(datadir, dset_name + inpf + '.npy')
            maxlabel = np.load(filename)
            maxlabel = maxlabel[0]
            print("read maxlabel from file")
        except IOError:
            maxlabel = np.amax(f['stack'][:,:,:])
            print("retrieved maxlabel from stack")

        fws = {}
        for propname in map_propnames:
            fws[propname] = np.zeros(maxlabel + 1)

        go2D = ((max_eccentricity is not None) or
                (min_solidity is not None) or 
                (min_euler_number is not None))
        if go2D:

            for i in range(0, f['stack'].shape[slicedim]):
                print("processing slice %d" % i)

                # TODO: mpi4py
                if slicedim == 0:
                    labels = f['stack'][i,:,:]
                    MBslc = fmb[maskMB[1]][i,:,:].astype('bool')
                elif slicedim == 1:
                    labels = f['stack'][:,i,:]
                    MBslc = fmb[maskMB[1]][:,i,:].astype('bool')
                elif slicedim == 2:
                    labels = f['stack'][:,:,i]
                    MBslc = fmb[maskMB[1]][:,:,i].astype('bool')

                fws = check_constraints(labels, fws, map_propnames,
                                        min_area, max_area,
                                        MBslc, max_intensity_mb,
                                        max_eccentricity,
                                        min_solidity,
                                        min_euler_number,
                                        min_extent)

        else:

            fws = check_constraints(f['stack'], fws, map_propnames,
                                    min_area, max_area,
                                    fmb[maskMB[1]], max_intensity_mb,
                                    max_eccentricity,
                                    min_solidity,
                                    min_euler_number,
                                    min_extent)

        for propname in map_propnames:
            filename = os.path.join(datadir, out + propname + '.npy')
            np.save(filename, fws[propname])

        f.close()
        fmb.close()

    elif mode == "2Dprops":

        out = dset_name + outpf + '_'

        filename = os.path.join(datadir, dset_name + inpf + '.h5')
        f = h5py.File(filename, 'r')

        fws = {}
        for propname in map_propnames:
            print("processing prop %s" % propname)

            filename = os.path.join(datadir, out + propname + '.npy')
            fws[propname] = np.load(filename)

            filename = os.path.join(datadir, out + propname + '.h5')
            g = h5py.File(filename, 'w')
            outds = g.create_dataset('stack', f['stack'].shape,
                                     dtype=fws[propname].dtype,
                                     compression="gzip")
            outds[:,:,:] = fws[propname][f['stack'][:,:,:]]
            g.close()

        f.close()

    elif mode == "2Dto3Dlabel":

        out = dset_name + outpf + '_'

        filename = os.path.join(datadir, out + 'label.h5')
        f = h5py.File(filename, 'r')

        filename = os.path.join(datadir, out + '3Dlabeled.h5')
        g = h5py.File(filename, 'w')
        outds = g.create_dataset('stack', f['stack'].shape,
                                 dtype='uint32',
                                 compression="gzip")
#         outds[:,:,:] = label(f['stack'][:,:,:] != 0)
        # scipy appears to have much less memory consumption
        outds[:,:,:] = scipy_label(f['stack'][:,:,:] != 0)[0]

        f.close()
        g.close()


# ========================================================================== #
# function defs
# ========================================================================== #


def check_constraints(labels, fws, propnames,
                      min_size=None, max_size=None,
                      MB=None, max_intensity_mb=None,
                      max_eccentricity=None,
                      min_solidity=None,
                      min_euler_number=None,
                      min_extent=None):
    """Compose forward maps according to label validity criteria."""

    rp = regionprops(labels, intensity_image=MB, cache=True)

    for prop in rp:

        if min_size is not None:
            if prop.area < min_size:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if max_size is not None:
            if prop.area > max_size:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if max_intensity_mb is not None:
            if prop.mean_intensity > max_intensity_mb:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if max_eccentricity is not None:
            if prop.eccentricity > max_eccentricity:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if min_solidity is not None:
            if prop.solidity < min_solidity:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if min_euler_number is not None:
            if prop.euler_number < min_euler_number:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if min_extent is not None:
            if prop.extent < min_extent:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue

        fws = set_fws(fws, prop, propnames, is_valid=True)

    for propname in propnames:
        if np.array(prop[propname]).dtype == 'int64':
            datatype = 'int32'
        else:
            datatype='float'

        fws[propname] = np.array(fws[propname], dtype=datatype)

    return fws


def set_fws(fws, prop, propnames, is_valid=False):
    """Set the forward maps entries for single labels."""

    for propname in propnames:
        if is_valid:
            fws[propname][prop.label] = prop[propname]
        else:
            fws[propname][prop.label] = 0
            # FIXME: for many prop '0' is a valid value (nan?)

    return fws


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
