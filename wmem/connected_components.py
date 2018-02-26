#!/usr/bin/env python

"""Label connected components.

"""

import sys
import argparse
import os

import numpy as np
from scipy.ndimage.measurements import label as scipy_label
from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects, binary_dilation
from skimage.measure import label, regionprops
try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")

from wmem import parse, utils


def main(argv):
    """Label connected components."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_connected_components(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    if args.mode == '3D':

        CC_3D(
            args.inputfile,
            args.maskDS,
            args.min_size_maskMM,
            args.min_area,
            args.outputfile,
            args.protective,
            )

    elif args.mode == '2D':

        CC_2D(
            args.inputfile,
            args.maskDS,
            args.slicedim,
            args.usempi & ('mpi4py' in sys.modules),
            args.outputfile,
            args.protective,
            )

    elif args.mode == "2Dfilter":

        criteria = (
            args.min_area,
            args.max_area,
            args.max_intensity_mb,
            args.max_eccentricity,
            args.min_solidity,
            args.min_euler_number,
            args.min_extent
            )

        CC_2Dfilter(
            args.inputfile,
            args.map_propnames,
            criteria,
            args.maskMB,
            args.slicedim,
            args.usempi & ('mpi4py' in sys.modules),
            args.outputfile,
            args.protective,
            )

    elif args.mode == "2Dprops":

        CC_2Dprops(
            args.inputfile,
            args.basename,
            args.map_propnames,
            args.usempi & ('mpi4py' in sys.modules),
            args.outputfile,
            args.protective,
            )

    elif args.mode == "2Dto3D":

        CC_2Dto3D(
            args.inputfile,
            args.outputfile,
            args.protective,
            )


# ========================================================================== #
# function defs
# ========================================================================== #


def CC_3D(
        h5path_in,
        h5path_mask='',
        min_size_maskMM=0,
        min_area=0,
        h5path_out='',
        protective=False,
        ):
    """Label connected components in a 3D stack."""

    # check output path
    if '.h5' in h5path_out:
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # open data for reading
    h5file_mm, ds_mm, elsize, axlab = utils.h5_load(h5path_in)
    if h5path_mask:
        h5file_md, ds_md, _, _ = utils.h5_load(h5path_mask)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_mm.shape, 'int32',
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # 3D labeling with label size constraints
    # NOTE: could save memory here by applying the constraints to input before
    if h5path_mask:
        mask = np.logical_or(binary_dilation(ds_mm[:]), ~ds_md[:])
    else:
        mask = binary_dilation(ds_mm[:])

    if min_size_maskMM:
        remove_small_objects(mask, min_size_maskMM, in_place=True)

    labels = label(~mask, return_num=False, connectivity=None)

    if min_area:
        remove_small_objects(labels, min_area, in_place=True)

    # remove the largest label (assumed unmyelinated axon compartment)
    rp = regionprops(labels)
    areas = [prop.area for prop in rp]
    labs = [prop.label for prop in rp]
    llab = labs[np.argmax(areas)]
    labels[labels == llab] = 0

    labels = relabel_sequential(labels)[0]

    ds_out[:] = labels

    # close and return
    try:
        h5file_mm.close()
        h5file_out.close()
        if h5path_mask:
            h5file_md.close()
    except (ValueError, AttributeError):
        return labels


def CC_2D(
        h5path_in,
        h5path_mask='',
        slicedim=0,
        usempi=False,
        h5path_out='',
        protective=False,
        ):
    """Label connected components in all slices."""

    # check output path
    if '.h5' in h5path_out:
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # open data for reading
    h5file_mm, ds_mm, elsize, axlab = utils.h5_load(h5path_in)
    if h5path_mask:
        h5file_md, ds_md, _, _ = utils.h5_load(h5path_mask)

    # prepare mpi  # TODO: could allow selection of slices/subset here
    n_slices = ds_mm.shape[slicedim]
    series = np.array(range(0, n_slices), dtype=int)
    if usempi:
        mpi_info = utils.get_mpi_info()
        series = utils.scatter_series(mpi_info, series)[0]
        comm = mpi_info['comm']
        rank = mpi_info['rank']
        size = mpi_info['size']
    else:
        comm = None
        rank = 0
        size = 1

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_mm.shape, 'uint32',
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab,
                                        usempi=usempi, comm=comm)

    # slicewise labeling
    maxlabel = 0
    for i in series:

        slcMM = utils.get_slice(ds_mm, i, slicedim, 'bool')
        if h5path_mask:
            slcMD = utils.get_slice(ds_md, i, slicedim, 'bool')
            labels, num = label(np.logical_and(~slcMM, slcMD), return_num=True)
        else:
            labels, num = label(~slcMM, return_num=True)
        print("found %d labels in slice %d" % (num, i))

        if usempi:
            # NOTE: assumed max number of labels in slice is 10000
            labels[~slcMM] += 10000 * i
            if i == n_slices - 1:
                maxlabel = np.amax(labels)
        else:
            labels[~slcMM] += maxlabel
            maxlabel += num

        if slicedim == 0:
            ds_out[i, :, :] = labels
        elif slicedim == 1:
            ds_out[:, i, :] = labels
        elif slicedim == 2:
            ds_out[:, :, i] = labels

    # save the maximum labelvalue in the dataset
    if usempi & (rank == size - 1):
        root = h5path_out.split('.h5')[0]
        fpath = root + '.npy'
        np.save(fpath, np.array([maxlabel]))

    # close and return
    try:
        h5file_mm.close()
        h5file_out.close()
        if h5path_mask:
            h5file_md.close()
    except (ValueError, AttributeError):
        return ds_out


def CC_2Dfilter(
        h5path_labels,
        map_propnames,
        criteria,
        h5path_int='',
        slicedim=0,
        usempi=False,
        outputfile='',
        protective=False,
        ):
    """Get forward mapping of labels/properties filtered by criteria."""

    (min_area,
     max_area,
     max_intensity_mb,
     max_eccentricity,
     min_solidity,
     min_euler_number,
     min_extent) = criteria

    # TODO: check output path

    # open data for reading
    h5file_mm, ds_mm, _, _ = utils.h5_load(h5path_labels)
    if h5path_int:
        h5file_mb, ds_mb, _, _ = utils.h5_load(h5path_int)
    else:
        ds_mb = None
    # mask used as intensity image in mean_intensity criterium

    # get the maximum labelvalue in the input
    root = h5path_labels.split('.h5')[0]
    maxlabel = get_maxlabel(root, ds_mm)

    # prepare mpi
    n_slices = ds_mm.shape[slicedim]
    series = np.array(range(0, n_slices), dtype=int)
    if usempi:
        mpi_info = utils.get_mpi_info()
        series = utils.scatter_series(mpi_info, series)[0]
        comm = mpi_info['comm']
        rank = mpi_info['rank']

        if rank == 0:
            fws_reduced = np.zeros((maxlabel, len(map_propnames)),
                                   dtype='float')
        else:
            fws_reduced = None
    else:
        comm = None
        rank = 0

    fws = np.zeros((maxlabel + 1, len(map_propnames)),
                   dtype='float')

    mapall = criteria.count(None) == len(criteria)

    # pick labels observing the constraints
    go2D = ((max_eccentricity is not None) or
            (min_solidity is not None) or
            (min_euler_number is not None) or
            mapall)
    if go2D:

        for i in series:
            slcMM = utils.get_slice(ds_mm, i, slicedim)
            if h5path_int:
                slcMB = utils.get_slice(ds_mb, i, slicedim)  # , 'bool'
            else:
                slcMB = None
            fws = check_constraints(slcMM, fws, map_propnames,
                                    criteria, slcMB, mapall)
        if usempi:  # FIXME
            comm.Reduce(fws, fws_reduced, op=MPI.MAX, root=0)
        else:
            fws_reduced = fws

    else:

        if rank == 0:
            fws = check_constraints(ds_mm, fws, map_propnames,
                                    criteria, ds_mb, mapall)
            fws_reduced = fws

    # write the forward maps to a numpy vector
    if rank == 0:
        slc = int(n_slices/2)
        slcMM = ds_mm[slc, :, :]
        slcMB = ds_mb[slc, :, :] if h5path_int else None
        datatypes = get_prop_datatypes(slcMM, map_propnames, slcMB)
        for i, propname in enumerate(map_propnames):
            root = outputfile.split('.h5')[0]
            nppath = '{}_{}.npy'.format(root, propname)
            outarray = np.array(fws_reduced[:, i], dtype=datatypes[i])
            np.save(nppath, outarray)

    # close and return
    h5file_mm.close()
    if h5path_int:
        h5file_mb.close()

    if rank == 0:
        return outarray


def CC_2Dprops(
        h5path_labels,
        basename,
        map_propnames,
        usempi=False,
        h5path_out='',
        protective=False,
        ):
    """Map the labels/properties."""

    # check output paths
    if '.h5' in h5path_out:
        for propname in map_propnames:
            h5path_prop = os.path.join(h5path_out, propname)
            status, info = utils.h5_check(h5path_out, protective)
            print(info)
            if status == "CANCELLED":
                return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_labels)

    # prepare mpi
    n_props = len(map_propnames)
    series = np.array(range(0, n_props), dtype=int)
    if usempi:
        mpi_info = utils.get_mpi_info()
        series = utils.scatter_series(mpi_info, series)[0]
        comm = mpi_info['comm']
    else:
        comm = None

    fws = {}
    for i in series:
        propname = map_propnames[i]
        print("processing prop %s" % propname)

        nppath = '{}_{}.npy'.format(basename, propname)
        fws[propname] = np.load(nppath)

        # open data for writing
        h5path_prop = os.path.join(h5path_out, propname)
        h5file_prop, ds_prop = utils.h5_write(None, ds_in.shape,
                                              fws[propname].dtype,
                                              h5path_prop,
                                              element_size_um=elsize,
                                              axislabels=axlab,
                                              usempi=usempi, comm=comm)

        ds_prop[:, :, :] = fws[propname][ds_in[:, :, :]]

        h5file_prop.close()

    # close and return
    h5file_in.close()


def CC_2Dto3D(
        h5path_in,
        h5path_out='',
        protective=False,
        ):
    """Label connected components in 3D from the 2D-generated mask."""

    # check output path
    if h5path_out:
        status, info = utils.h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, 'uint32',
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    # NOTE: scipy has much lower memory consumption than scikit-image
    ds_out[:] = scipy_label(ds_in[:, :, :] != 0)[0]

    # close and return
    try:
        h5file_in.close()
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def check_constraints(labels, fws, propnames, criteria, MB=None, mapall=False):
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

        if mapall:
            fws = set_fws(fws, prop, propnames, is_valid=True)
            continue

        # TODO: ordering
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


def get_prop_datatypes(labels, propnames, MB=None):
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
        if (propname == 'label') and (not is_valid):
            fws[prop.label, i] = 0
            # FIXME: for many prop '0' is a valid value (nan?)
        else:
            fws[prop.label, i] = prop[propname]
#         if is_valid:  #  or (propname == 'orig')
#             fws[prop.label, i] = prop[propname]
#         else:
#             fws[prop.label, i] = 0
#             # FIXME: for many prop '0' is a valid value (nan?)

    return fws


def get_maxlabel(root, ds):
    """Read the maximum label value from file or retrieve from array."""

    try:
        maxlabel = np.load(root + '.npy')
        maxlabel = maxlabel[0]
        print("read maxlabel from file: {}".format(maxlabel))
    except IOError:
        maxlabel = np.amax(ds[:, :, :])
        print("retrieved maxlabel from dataset: {}".format(maxlabel))

    return maxlabel


if __name__ == "__main__":
    main(sys.argv[1:])
