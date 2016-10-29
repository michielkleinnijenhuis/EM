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


# TODO: write elsize and axislabels
def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-i', '--inpf', nargs=2,
                        default=['_labelMA_2Dcore', 'stack'],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMA_2Dcore_fw_', 'stack'],
                        help='...')
    parser.add_argument('-b', '--basename', default='',
                        help='...')
    parser.add_argument('--maskDS', nargs=2, default=['_maskDS', 'stack'],
                        help='...')
    parser.add_argument('--maskMM', nargs=2, default=['_maskMM', 'stack'],
                        help='...')
    parser.add_argument('--maskMB', nargs=2, default=['_maskMB', 'stack'],
                        help='...')
    parser.add_argument('-d', '--slicedim', type=int, default=0,
                        help='...')

    parser.add_argument('-M', '--mode',
                        help='...')
    parser.add_argument('-p', '--map_propnames', nargs='*',
                        help='...')

    parser.add_argument('-q', '--min_size_maskMM', type=int, default=None,
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
    if args.basename:
        basename = args.basename
    else:
        basename = dset_name
    inpf = args.inpf
    outpf = args.outpf
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMB = args.maskMB
    mode = args.mode
    slicedim = args.slicedim
    map_propnames = args.map_propnames
    min_size_maskMM = args.min_size_maskMM
    min_area = args.min_area
    max_area = args.max_area
    max_intensity_mb = args.max_intensity_mb
    max_eccentricity = args.max_eccentricity
    min_euler_number = args.min_euler_number
    min_solidity = args.min_solidity
    min_extent = args.min_extent
    usempi = args.usempi & ('mpi4py' in sys.modules)

    if mode == '3D':

        CC_3D(datadir, dset_name, maskDS, maskMM,
              min_size_maskMM, min_area, outpf)

    elif mode == '2D':

        CC_2D(datadir, dset_name, maskDS, maskMM,
              slicedim, usempi, outpf)

    elif mode == "2Dfilter":

        criteria = (min_area,
                    max_area,
                    max_intensity_mb,
                    max_eccentricity,
                    min_solidity,
                    min_euler_number,
                    min_extent)

        CC_filter2D(datadir, dset_name, inpf, maskMB,
                    map_propnames, criteria,
                    slicedim, usempi, outpf)

    elif mode == "2Dprops":

        CC_props2D(datadir, dset_name, inpf, basename,
                   map_propnames, usempi, outpf)

    elif mode == "2Dto3Dlabel":

        CC_label2Dto3D(datadir, dset_name, inpf, outpf)


# ========================================================================== #
# function defs
# ========================================================================== #


def CC_3D(datadir, dset_name, maskDS, maskMM,
          min_size_maskMM, min_area, outpf):
    """Label connected components in a 3D stack."""

    maskDS, elsize, al = loadh5(datadir, dset_name + maskDS[0],
                                fieldname=maskDS[1], dtype='bool')
    maskMM = loadh5(datadir, dset_name + maskMM[0],
                    fieldname=maskMM[1], dtype='bool')[0]

    mask = np.logical_or(binary_dilation(maskMM), ~maskDS)

    if min_size_maskMM is not None:
        remove_small_objects(mask, min_size_maskMM, in_place=True)

    labels = label(~mask, return_num=False, connectivity=None)

    if min_area is not None:
        remove_small_objects(labels, min_area, in_place=True)

    # remove the unmyelinated axons (largest label)
    rp = regionprops(labels)
    areas = [prop.area for prop in rp]
    labs = [prop.label for prop in rp]
    llab = labs[np.argmax(areas)]
    labels[labels == llab] = 0

    labels = relabel_sequential(labels)[0]

    writeh5(labels, datadir, dset_name + outpf[0], outpf[1],
            dtype='int32', element_size_um=elsize, axislabels=al)


def CC_2D(datadir, dset_name, maskDS, maskMM,
          slicedim, usempi, outpf):
    """Label connected components in all slices."""

    gname = os.path.join(datadir, dset_name + outpf[0] + '.h5')
    dsname = os.path.join(datadir, dset_name + maskDS[0] + '.h5')
    mmname = os.path.join(datadir, dset_name + maskMM[0] + '.h5')

    if usempi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        g = h5py.File(gname, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        ds = h5py.File(dsname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        dstack = ds[maskDS[1]]
        mm = h5py.File(mmname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        mstack = mm[maskMM[1]]

        n_slices = mstack.shape[slicedim]
        local_nrs = scatter_series(n_slices, comm, size, rank,
                                   MPI.SIGNED_LONG_LONG)[0]
    else:
        g = h5py.File(gname, 'w')
        ds = h5py.File(dsname, 'r')
        dstack = ds[maskDS[1]]
        mm = h5py.File(mmname, 'r')
        mstack = mm[maskMM[1]]

        n_slices = mstack.shape[slicedim]
        local_nrs = np.array(range(0, n_slices), dtype=int)

    outds = g.create_dataset(outpf[1], mstack.shape,
                             dtype='uint32',
                             compression=None if usempi else 'gzip')
    elsize, al = get_h5_attributes(mstack)
    write_h5_attributes(g[outpf[1]], elsize, al)

    maxlabel = 0
    for i in local_nrs:

        if slicedim == 0:
            DSslc = dstack[i, :, :].astype('bool')
            MMslc = mstack[i, :, :].astype('bool')
        elif slicedim == 1:
            DSslc = dstack[:, i, :].astype('bool')
            MMslc = mstack[:, i, :].astype('bool')
        elif slicedim == 2:
            DSslc = dstack[:, :, i].astype('bool')
            MMslc = mstack[:, :, i].astype('bool')

        labels, num = label(np.logical_and(~MMslc, DSslc), return_num=True)
        print("found %d labels in slice %d" % (num, i))
        if usempi:
            # NOTE: assumed max number of labels in slice is 10000
            labels[~MMslc] += 10000 * i
            if i == n_slices - 1:
                maxlabel = np.amax(labels)
        else:
            labels[~MMslc] += maxlabel
            maxlabel += num

        if slicedim == 0:
            outds[i, :, :] = labels
        elif slicedim == 1:
            outds[:, i, :] = labels
        elif slicedim == 2:
            outds[:, :, i] = labels

    if usempi & (rank == size - 1):
        filename = os.path.join(datadir, dset_name + outpf[0] + '.npy')
        np.save(filename, np.array([maxlabel]))

    g.close()
    mm.close()
    ds.close()


def CC_filter2D(datadir, dset_name, inpf, maskMB,
                map_propnames, criteria,
                slicedim, usempi, outpf):
    """Get forward mapping of labels/properties filtered by criteria."""

    (min_area,
     max_area,
     max_intensity_mb,
     max_eccentricity,
     min_solidity,
     min_euler_number,
     min_extent) = criteria

    fname = os.path.join(datadir, dset_name + inpf[0] + '.h5')
    mbname = os.path.join(datadir, dset_name + maskMB[0] + '.h5')

    if usempi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        f = h5py.File(fname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        fstack = f[inpf[1]]
        mb = h5py.File(mbname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        mbstack = mb[maskMB[1]]

        n_slices = fstack.shape[slicedim]
        local_nrs = scatter_series(n_slices, comm, size, rank,
                                   MPI.SIGNED_LONG_LONG)[0]

        maxlabel = get_maxlabel(datadir, dset_name + inpf[0], fstack)

        if rank == 0:
            fws_reduced = np.zeros((maxlabel, len(map_propnames)),
                                   dtype='float')
        else:
            fws_reduced = None

    else:
        rank = 0

        f = h5py.File(fname, 'r')
        fstack = f[inpf[1]]
        mb = h5py.File(mbname, 'r')
        mbstack = mb[maskMB[1]]

        n_slices = fstack.shape[slicedim]
        local_nrs = np.array(range(0, n_slices), dtype=int)

        maxlabel = get_maxlabel(datadir, dset_name + inpf[1], fstack)

    fws = np.zeros((maxlabel + 1, len(map_propnames)),
                   dtype='float')

    go2D = ((max_eccentricity is not None) or
            (min_solidity is not None) or
            (min_euler_number is not None))
    if go2D:

        for i in local_nrs:

            if slicedim == 0:
                labels = fstack[i, :, :]
                MBslc = mbstack[i, :, :].astype('bool')
            elif slicedim == 1:
                labels = fstack[:, i, :]
                MBslc = mbstack[:, i, :].astype('bool')
            elif slicedim == 2:
                labels = fstack[:, :, i]
                MBslc = mbstack[:, :, i].astype('bool')

            fws = check_constraints(labels, fws, map_propnames,
                                    criteria, MBslc)

        if usempi:
            # FIXME
            comm.Reduce(fws, fws_reduced, op=MPI.MAX, root=0)
        else:
            fws_reduced = fws

    else:
        if rank == 0:
            fws = check_constraints(fstack, fws, map_propnames,
                                    criteria, mbstack)
            fws_reduced = fws

    if rank == 0:
        slc = int(n_slices/2)
        datatypes = get_prop_datatypes(fstack[:, :, slc],
                                       mbstack[:, :, slc],
                                       map_propnames)
        for i, propname in enumerate(map_propnames):
            filename = dset_name + outpf[0] + '_' + propname + '.npy'
            filepath = os.path.join(datadir, filename)
            outarray = np.array(fws_reduced[:, i], dtype=datatypes[i])
            np.save(filepath, outarray)

    f.close()
    mb.close()


def CC_props2D(datadir, dset_name, inpf, basename,
               map_propnames, usempi, outpf):
    """Map the labels/properties."""

    fname = os.path.join(datadir, dset_name + inpf[0] + '.h5')

    if usempi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        f = h5py.File(fname, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        fstack = f[inpf[1]]

        local_nrs = scatter_series(len(map_propnames), comm, size, rank,
                                   MPI.SIGNED_LONG_LONG)[0]
    else:
        f = h5py.File(fname, 'r')
        fstack = f[inpf[1]]

        local_nrs = np.array(range(0, len(map_propnames)), dtype=int)

    fws = {}
    for i in local_nrs:
        propname = map_propnames[i]
        print("processing prop %s" % propname)

        filename = basename + outpf[0] + '_' + propname + '.npy'
        filepath = os.path.join(datadir, filename)
        fws[propname] = np.load(filepath)

        gname = dset_name + outpf[0] + '_' + propname + '.h5'
        gpath = os.path.join(datadir, gname)
        g = h5py.File(gpath, 'w')
        outds = g.create_dataset(outpf[1], fstack.shape,
                                 dtype=fws[propname].dtype,
                                 compression=None if usempi else 'gzip')
        elsize, al = get_h5_attributes(fstack)
        write_h5_attributes(g[outpf[1]], elsize, al)

        outds[:, :, :] = fws[propname][fstack[:, :, :]]
        g.close()

    f.close()


def CC_label2Dto3D(datadir, dset_name, inpf, outpf):
    """Label connected components in 3D from the 2D-generated mask."""

    fname = os.path.join(datadir, dset_name + inpf[0])
    f = h5py.File(fname, 'r')
    fstack = f[inpf[1]]

    gname = os.path.join(datadir, dset_name + outpf[0])
    g = h5py.File(gname, 'w')
    outds = g.create_dataset(outpf[1], fstack.shape,
                             dtype='uint32',
                             compression="gzip")
    elsize, al = get_h5_attributes(fstack)
    write_h5_attributes(g[outpf[1]], elsize, al)
    # NOTE:scipy appears to have much less memory consumption
#         gstack[:, :, :] = label(fstack[:, :, :] != 0)
    outds[:, :, :] = scipy_label(fstack[:, :, :] != 0)[0]

    f.close()
    g.close()


def check_constraints(labels, fws, propnames, criteria, MB=None):
    """Compose forward maps according to label validity criteria."""

    (min_area,
     max_area,
     max_intensity_mb,
     max_eccentricity,
     min_solidity,
     min_euler_number,
     min_extent) = criteria

    rp = regionprops(labels, intensity_image=MB, cache=True)

    for prop in rp:

        if min_area is not None:
            if prop.area < min_area:
                fws = set_fws(fws, prop, propnames, is_valid=False)
                continue
        if max_area is not None:
            if prop.area > max_area:
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

    return fws


def get_prop_datatypes(labels, MB, propnames):
    """Retrieve the per-property output datatypes."""

    rp = regionprops(labels, intensity_image=MB, cache=True)
    datatypes = []
    for propname in propnames:
        if np.array(rp[0][propname]).dtype == 'int64':
            datatypes.append('int32')
        else:
            datatypes.append('float')

    return datatypes


def set_fws(fws, prop, propnames, is_valid=False):
    """Set the forward maps entries for single labels."""

    for i, propname in enumerate(propnames):
        if is_valid:
            fws[prop.label, i] = prop[propname]
        else:
            fws[prop.label, i] = 0
            # FIXME: for many prop '0' is a valid value (nan?)

    return fws


def get_maxlabel(datadir, fstem, fstack):
    """Read the maximum label value from file or retrieve from array."""

    try:
        filename = os.path.join(datadir, fstem + '.npy')
        maxlabel = np.load(filename)
        maxlabel = maxlabel[0]
        print("read maxlabel from file")
    except IOError:
        maxlabel = np.amax(fstack[:, :, :])
        print("retrieved maxlabel from stack")

    return maxlabel


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
    """Load a h5 stack."""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]

    element_size_um, axislabels = get_h5_attributes(f[fieldname])

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """Write a h5 stack."""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    write_h5_attributes(g[fieldname], element_size_um, axislabels)

    g.close()


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
