#!/usr/bin/env python

"""General utility functions for the wmem package.

"""

import os
import sys
import importlib
import errno
import pickle
import re
from random import shuffle
import socket
import glob

import numpy as np

sys.stdout = open(os.devnull, 'w')

try:
    from skimage.io import imread, imsave
except ImportError:
    print("scikit image could not be loaded")


try:
    import nibabel as nib
except ImportError:
    print("nibabel could not be loaded")

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

sys.stdout = sys.__stdout__

import h5py

from wmem import Image, LabelImage, MaskImage


def mkdir_p(filepath):
    try:
        os.makedirs(filepath)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(filepath):
            pass
        else:
            raise


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
    parts = re.findall('([0-9]{5}-[0-9]{5})', fname)
    id_string = '_'.join(parts)
    dset_name = fname.split(id_string)[0][:-1]
    
    x = int(parts[-3].split("-")[0]) - blockoffset[0]
    X = int(parts[-3].split("-")[1]) - blockoffset[0]
    y = int(parts[-2].split("-")[0]) - blockoffset[1]
    Y = int(parts[-2].split("-")[1]) - blockoffset[1]
    z = int(parts[-1].split("-")[0]) - blockoffset[2]
    Z = int(parts[-1].split("-")[1]) - blockoffset[2]

    dset_info = {'datadir': datadir, 'base': dset_name,
                 'nzfills': len(parts[1].split("-")[0]),
                 'postfix': id_string,
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}

    return dset_info, x, X, y, Y, z, Z


def dset_name2slices(dset_name, blockoffset, axlab='xyz', shape=[]):
    """Get slices from data indices in a filename."""

    _, x, X, y, Y, z, Z = split_filename(dset_name, blockoffset)
    slicedict = {'x': slice(x, X, 1),
                 'y': slice(y, Y, 1),
                 'z': slice(z, Z, 1)}
    for dim in ['c', 't']:
        if dim in axlab:
            upper = shape[axlab.index(dim)]
            slicedict[dim] = slice(0, upper, 1)

    slices = [slicedict[dim] for dim in axlab]

    return slices


def xyz_datarange(xyz, files):
    """Get the full ranges for x,y,z if upper bound is undefined."""

    firstimage = imread(files[0])

    X = xyz[1] or firstimage.shape[1]
    Y = xyz[3] or firstimage.shape[0]
    Z = xyz[5] or len(files)

    return xyz[0], X, xyz[2], Y, xyz[4], Z


def get_slice_objects(dataslices, dims):
    """Get the full ranges for z, y, x if upper bound is undefined."""

    # set default dataslices
    if dataslices is None:
        dataslices = []
        for dim in dims:
            dataslices += [0, dim, 1]

    starts = dataslices[::3]
    stops = dataslices[1::3]
    stops = [dim if stop == 0 else stop
             for stop, dim in zip(stops, dims)]
    steps = dataslices[2::3]
    slices = [slice(start, stop, step)
              for start, stop, step in zip(starts, stops, steps)]

    return slices


def get_slice_objects_prc(dataslices, zyxdims):
    """Get the full ranges for z, y, x if upper bound is undefined."""

    # set default dataslice
    if dataslices is None:
        dataslices = [0, 0, 1, 0, 0, 1, 0, 0, 1]

#     # derive the stop-values from the image data if not specified
#     if files[0].endswith('.dm3'):
#         try:
#             import DM3lib as dm3
#         except ImportError:
#             raise
#         dm3f = dm3.DM3(files[0], debug=0)
#         dim0, dim1 = tuple(dm3f.imagedata.shape)
# #         dim0 = dm3f.tags.get('root.ImageList.1.ImageData.Dimensions.0')
# #         dim1 = dm3f.tags.get('root.ImageList.1.ImageData.Dimensions.1')
#     else:
#         dim0, dim1 = tuple(imread(files[0]).shape)
# 
#     zyxdims[0] = len(files)
    dataslices[1] = dataslices[1] or zyxdims[0]
    dataslices[4] = dataslices[4] or zyxdims[1]
    dataslices[7] = dataslices[7] or zyxdims[2]

    # create the slice objects
    if dataslices is not None:
        starts = dataslices[::3]
        stops = dataslices[1::3]
        steps = dataslices[2::3]
        slices = [slice(start, stop, step)
                  for start, stop, step in zip(starts, stops, steps)]

    return slices


def slices2sizes(slices):

    return (len(range(*slc.indices(slc.stop))) for slc in slices)


def xyz_datashape(al, xyz):
    """Get datashape from axislabels and bounds."""

    x, X, y, Y, z, Z = xyz

    if al == 'zyx':
        datalayout = (Z - z, Y - y, X - x)
    elif al == 'zxy':
        datalayout = (Z - z, X - x, Y - y)
    elif al == 'yzx':
        datalayout = (Y - y, Z - z, X - x)
    elif al == 'yxz':
        datalayout = (Y - y, X - x, Z - z)
    elif al == 'xzy':
        datalayout = (X - x, Z - z, Y - y)
    else:
        datalayout = (X - x, Y - y, Z - z)

    return datalayout


def get_mpi_info(usempi=False, mpi_dtype=''):
    """Get an MPI communicator."""

    if usempi:

        if not mpi_dtype:
            mpi_dtype = MPI.SIGNED_LONG_LONG
        comm = MPI.COMM_WORLD

        mpi_info = {
            'enabled': True,
            'comm': comm,
            'rank': comm.Get_rank(),
            'size': comm.Get_size(),
            'dtype': mpi_dtype
            }

    else:

        mpi_info = {
            'enabled': False,
            'comm': None,
            'rank': 0,
            'size': 1,
            'dtype': None,
            }

    return mpi_info


def scatter_series(mpi_info, nblocks):
    """Scatter a series of jobnrs over processes."""

    series = np.array(range(0, nblocks), dtype=int)

    if not mpi_info['enabled']:
        return series, None, None

    comm = mpi_info['comm']
    rank = mpi_info['rank']
    size = mpi_info['size']

    n_all = len(series)
    n_local = np.ones(size, dtype=int) * n_all / size
    n_local[0:n_all % size] += 1

    series = np.array(series, dtype=int)
    series_local = np.zeros(n_local[rank], dtype=int)

    displacements = tuple(sum(n_local[0:r]) for r in range(0, size))

    comm.Scatterv([series, tuple(n_local), displacements, mpi_info['dtype']],
                  series_local, root=0)

    return series_local, tuple(n_local), displacements


def output_check(outpaths, save_steps=True, protective=False):
    """Check output paths for writing."""

    if not outpaths['out']:
        status = "WARNING"
        info = "not writing results to file"
        print("{}: {}".format(status, info))
        return status

    # validate any additional output formats
    if 'addext' in outpaths.keys():
        root = outpaths['addext'][0]
        for ext in outpaths['addext'][1]:
            status = output_check_all(root, ext, None,
                                      save_steps, protective)
            if status == "CANCELLED":
                return status

    # validate the main output
    root, ext = os.path.splitext(outpaths['out'])
    status = output_check_all(root, ext, outpaths,
                              save_steps, protective)

    return status


def output_check_all(root, ext, outpaths=None, save_steps=False, protective=False):

    if '.h5' in ext:
        if outpaths is None:
            status = output_check_h5(['{}{}'.format(root, ext)], save_steps, protective)
        else:
            status = output_check_h5(outpaths, save_steps, protective)
    elif '.nii' in ext:
        status = output_check_dir(['{}{}'.format(root, ext)], protective)
    else:  # directory with images assumed
        status = output_check_dir([root], protective)

    return status


def output_check_h5(outpaths, save_steps=True, protective=False):

    try:
        root, ds_main = outpaths['out'].split('.h5')
        h5file_out = h5py.File(root + '.h5', 'a')

    except ValueError:
        status = "CANCELLED"
        info = "main output is not a valid h5 dataset"
        print("{}: {}".format(status, info))
        return status

    else:
        # create a group and set outpath for any intermediate steps
        for dsname, outpath in outpaths.items():
            if ((dsname != 'out') and save_steps and (not outpath)):
                grpname = ds_main + "_steps"
                try:
                    h5file_out[grpname]
                except KeyError:
                    h5file_out.create_group(grpname)
                outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)

        h5file_out.close()

        # check the path for each h5 output
        for _, outpath in outpaths.items():
            if outpath:
                status, info = h5_check(outpath, protective)
                print("{}: {}".format(status, info))
                if status == "CANCELLED":
                    return status


def output_check_dir(outpaths, protective):
    """Check output paths for writing."""

    status = ''
    for outpath in outpaths:
        if os.path.exists(outpath):
            if protective:
                status = 'CANCELLED'
                info = "protecting {}".format(outpath)
                print("{}: {}".format(status, info))
                return status
            else:
                status = "WARNING"
                info = 'overwriting {}'.format(outpath)
                print("{}: {}".format(status, info))
    if not status:
        outdir = os.path.dirname(outpaths[0])
        status = "INFO"
        info = "writing to {}".format(outdir)
        print("{}: {}".format(status, info))

    return status


def h5_check(h5path_full, protective=False):
    """Check if dataset exists in a h5 file."""

    basepath, h5path_dset = h5path_full.split('.h5')
    h5path_file = basepath + '.h5'

    try:
        h5file = h5py.File(h5path_file, 'r+')
    except IOError:  # FIXME: it's okay for the file not to be there
        status = "INFO"
        info = "could not open {}".format(h5path_file)
#             info = "{}: writing to {}".format(status, h5path_full)
    else:
        if h5path_dset in h5file:
            if protective:  # TODO: raise error
                status = "CANCELLED"
                info = "protecting {}".format(h5path_full)
            else:
                status = "WARNING"
                info = "overwriting {}".format(h5path_full)
        else:
            status = "INFO"
            info = "writing to {}".format(h5path_full)

        h5file.close()

    return status, info


def load(dspath, load_data=False,
         dtype='', dataslices=None,
         inlayout=None, outlayout=None):
    """Load a dataset."""

    if not dspath:
        return None, None, None, None

    try:
        _, _ = dspath.split('.h5')
    except ValueError:
        pass
    else:
        return h5_load(dspath, load_data,
                       dtype, dataslices,
                       inlayout, outlayout)

    try:
        nib.load(dspath)
    except nib.filebasedimages.ImageFileError:
        pass
    except:
        raise
    else:
        return nii_load(dspath, load_data,
                        dtype, dataslices,
                        inlayout, outlayout)

    try:
        imread(dspath)
    except:
        pass
    else:
        return imf_load(dspath, load_data,
                        dtype, dataslices,
                        inlayout, outlayout)

#     try:
#         basepath, h5path_dset = dspath.split('.h5')
# #         h5path_file = basepath + '.h5'
# #         file_in = h5py.File(h5path_file, 'r+')
# #         file_in.close()
#         nib.load(dspath)
#     except ValueError:
#         fun_load = nii_load
#     except nib.filebasedimages.ImageFileError:
#         fun_load = h5_load
#     except:
#         raise
#     return fun_load(dspath, load_data, dtype, channels, inlayout, outlayout)


def imf_load(dspath, load_data=True,
             dtype='', dataslices=None,
             inlayout=None, outlayout=None):

    # FIXME: proper handling of inlayout etc

    ds_in = imread(dspath)
    elsize = np.array([1, 1])
    axlab = 'yxc'
    outlayout = outlayout or axlab

    data, elsize, axlab, slices = load_dataset(
        ds_in, elsize, axlab, outlayout, dtype, dataslices)

    return data, elsize, axlab, slices


def imf_write(dspath, data):

    imsave(dspath, data)


def nii_load(dspath, load_data=False,
             dtype='', dataslices=None,
             inlayout=None, outlayout=None):
    """Load a nifti dataset."""

    file_in = nib.load(dspath)

    ds_in = file_in.dataobj  # proxy

    elsize = list(file_in.header.get_zooms())
    ndim = len(elsize)
    axlab = 'xyztc'[:ndim]  # FIXME: get from header?
    axlab = inlayout or axlab or 'xyztc'[:ndim]

    if load_data:
        outlayout = outlayout or axlab
        data, elsize, axlab, slices = load_dataset(
            ds_in, elsize, axlab, outlayout, dtype, dataslices)

        return data, elsize, axlab, slices

    else:

        return file_in, ds_in, elsize, axlab


def h5_load(dspath, load_data=False,
            dtype='', dataslices=None,
            inlayout=None, outlayout=None,
            comm=None):
    """Load a h5 dataset."""

    basepath, h5path_dset = dspath.split('.h5')
    h5path_file = basepath + '.h5'
    if comm is not None:
        h5file = h5py.File(h5path_file, 'r+', driver='mpio', comm=comm)
    else:
        h5file = h5py.File(h5path_file, 'r')

    ds_in = h5file[h5path_dset]  # proxy

    try:
        ndim = ds_in.ndim
    except AttributeError:
        ndim = len(ds_in.dims)

    elsize, axlab = h5_load_attributes(ds_in)
    if elsize is None:
        elsize = np.array([1] * ndim)
    if inlayout is not None:
        axlab = inlayout
    elif axlab is None:
        axlab = 'zyxct'[:ndim]

    if load_data:
        outlayout = outlayout or axlab
        data, elsize, axlab, slices = load_dataset(
            ds_in, elsize, axlab, outlayout, dtype, dataslices)
        h5file.close()

        return data, elsize, axlab, slices

    return h5file, ds_in, elsize, axlab


def load_dataset(ds, elsize=[], axlab='',
                 outlayout='', dtype='',
                 dataslices=None, uint8conv=False):
    """Load data from a proxy and select/transpose/convert/...."""

    slices = get_slice_objects(dataslices, ds.shape)

    data = slice_dataset(ds, slices)

    if list(axlab) != list(outlayout):
        in2out = [axlab.index(l) for l in outlayout]
        data = np.transpose(data, in2out)
        elsize = np.array(elsize)[in2out]
        axlab = outlayout
        slices = [slices[i] for i in in2out]

    if dtype:
        data = data.astype(dtype, copy=False)

    if uint8conv:
        from skimage import img_as_ubyte
        data = normalize_data(data)[0]
        data = img_as_ubyte(data)

    return data, elsize, axlab, slices


def slice_dataset(ds, slices):

    try:
        ndim = ds.ndim
    except AttributeError:
        ndim = len(ds.dims)

    if ndim == 1:
        data = ds[slices[0]]
    elif ndim == 2:
        data = ds[slices[0], slices[1]]
    elif ndim == 3:
        data = ds[slices[0], slices[1], slices[2]]
    elif ndim == 4:
        data = ds[slices[0], slices[1], slices[2], slices[3]]
    elif ndim == 5:
        data = ds[slices[0], slices[1], slices[2], slices[3], slices[4]]

    return np.squeeze(data)


def ds_in2out(in2out, ds, elsize):
    """Transpose dataset and element sizes."""

    elsize = elsize[in2out]
    ds = np.transpose(ds, in2out)

    return ds, elsize


def h5_write(data, shape, dtype,
             h5path_full, h5file=None,
             element_size_um=None, axislabels=None,
             chunks=True, compression="gzip",
             comm=None,
             slices=None):
    """Write a h5 dataset."""

    if comm is not None:
        chunks = None
        compression = None

    if h5path_full:

        basepath, h5path_dset = h5path_full.split('.h5')
        if not isinstance(h5file, h5py.File):
            h5path_file = basepath + '.h5'
            if comm is not None:
                h5file = h5py.File(h5path_file, 'a',
                                   driver='mpio', comm=comm)
            else:
                h5file = h5py.File(h5path_file, 'a')

        if h5path_dset in h5file:
            h5ds = h5file[h5path_dset]
        else:
            if comm is not None:
                h5ds = h5file.create_dataset(h5path_dset,
                                             shape=shape,
                                             dtype=dtype)
            else:
                h5ds = h5file.create_dataset(h5path_dset,
                                             shape=shape,
                                             dtype=dtype,
                                             chunks=chunks,
                                             compression=compression)
            h5_write_attributes(h5ds, element_size_um, axislabels)

        if data is not None:

            write_to_h5ds(h5ds, data, slices)

            h5file.close()

        else:

            return h5file, h5ds

    else:

        return None, np.empty(shape, dtype)


def write_to_h5ds(h5ds, data, slices=None):
    """Write data to a hdf5 dataset."""

    if slices is None:

        h5ds[:] = data

    else:

        try:
            ndim = data.ndim
        except AttributeError:
            ndim = len(data.dims)

        if ndim == 1:
            h5ds[slices[0]] = data
        elif ndim == 2:
            h5ds[slices[0], slices[1]] = data
        elif ndim == 3:
            h5ds[slices[0], slices[1], slices[2]] = data
        elif ndim == 4:
            h5ds[slices[0], slices[1], slices[2], slices[3]] = data
        elif ndim == 5:
            h5ds[slices[0], slices[1], slices[2], slices[3], slices[4]] = data


def h5_load_attributes(h5ds):
    """Get attributes from a dataset."""

    element_size_um = axislabels = None

    if 'element_size_um' in h5ds.attrs.keys():
        element_size_um = h5ds.attrs['element_size_um']

    if 'DIMENSION_LABELS' in h5ds.attrs.keys():
        axislabels = h5ds.attrs['DIMENSION_LABELS']
    # FIXME: if this fails it empties the file!?
#     IOError: Unable to read attribute (Address of object past end of allocation)
    # GOOD
#     Attribute: DIMENSION_LABELS {3}
#         Type:      variable-length null-terminated ASCII string
#         Data:  "z", "y", "x"
    # BAD (matlab)
#     Attribute: DIMENSION_LABELS scalar
#         Type:      3-byte null-terminated ASCII string
#         Data:  "zyx"

    return element_size_um, axislabels


def h5_write_attributes(h5ds, element_size_um=None, axislabels=None):
    """Write attributes to a dataset."""

    if element_size_um is not None:
        h5ds.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, label in enumerate(axislabels):
            h5ds.dims[i].label = label


def string_masks(masks, mask, dataslices=None):
    """Work out a mask from a series of operations and masks."""

    if not masks:
        return mask

    aliases = {'NOT': 'np.logical_not',
               'AND': 'np.logical_and',
               'OR': 'np.logical_or',
               'XOR': 'np.logical_xor'}

    not_flag = False
    op = eval('np.logical_and')

    for m in masks:
        if m in aliases.keys():
            m = aliases[m]
            if eval(m) is np.logical_not:
                not_flag = True
            else:
                op = eval(m)
        else:
            newmask = h5_load(m, load_data=True, dtype='bool',
                              dataslices=dataslices)[0]
            if not_flag:
                np.logical_not(newmask, newmask)
                not_flag = False
            op(mask, newmask, mask)

    return mask


def write_to_nifti(filepath, data, element_size_um):
    """Write a dataset to nifti format."""

    # FIXME: more flexible transforms?!
    mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    if element_size_um is not None:
        mat[0][0] = element_size_um[0]
        mat[1][1] = element_size_um[1]
        mat[2][2] = element_size_um[2]

    if data.dtype == 'bool':
        data = data.astype('uint8')

    nib.Nifti1Image(data, mat).to_filename(filepath)


def write_to_img(basepath, data, outlayout, nzfills=5, ext='.png', slcoffset=0):
    """Write a 3D dataset to a stack of 2D image files."""

    mkdir_p(basepath)
    fstring = '{{:0{0}d}}'.format(nzfills)

    if ext != '.tif':
        data = normalize_data(data)[0]

    if data.ndim == 2:
        slcno = slcoffset
        filepath = os.path.join(basepath, fstring.format(slcno) + ext)
        imsave(filepath, data)

    for slc in range(0, data.shape[outlayout.index('z')]):
        slcno = slc + slcoffset
        if outlayout.index('z') == 0:
            slcdata = data[slc, :, :]
        elif outlayout.index('z') == 1:
            slcdata = data[:, slc, :]
        elif outlayout.index('z') == 2:
            slcdata = data[:, :, slc]

        filepath = os.path.join(basepath, fstring.format(slcno) + ext)
        imsave(filepath, slcdata)


def normalize_data(data):
    """Normalize data between 0 and 1."""

    data = data.astype('float64')
    datamin = np.amin(data)
    datamax = np.amax(data)
    data -= datamin
    data *= 1/(datamax-datamin)

    return data, [datamin, datamax]


def get_slice(ds, i, slicedim, datatype=''):
    """Retrieve a single slice from a 3D array."""

    if slicedim == 0:
        slc = ds[i, :, :]
    elif slicedim == 1:
        slc = ds[:, i, :]
    elif slicedim == 2:
        slc = ds[:, :, i]

    if datatype:
        slc = slc.astype(datatype)

    return slc


def classify_label_set(labelsets, labelset, lskey=None):
    """Add labels to a labelset or create a new set."""

    found = False
    for lsk in sorted(labelsets.keys()):
        lsv = labelsets[lsk]
        for l in labelset:
            if l in lsv:
                labelsets[lsk] = lsv | labelset
                found = True
                return labelsets
    if not found:
        if lskey is None:
            lskey = min(labelset)
        labelsets[lskey] = labelset

    return labelsets


def classify_label_list(MAlist, labelset):
    """Add set of labels to an axonset or create new axonset."""

    found = False
    for i, MA in enumerate(MAlist):
        for l in labelset:
            if l in MA:
                MAlist[i] = MA | labelset
                found = True
                break
    if not found:
        MAlist.append(labelset)

    return MAlist


def forward_map_list(fw, labels, MAlist):
    """Map all labelsets in MAlist to axons."""

    for MA in MAlist:
        MA = sorted(list(MA))
        for l in MA:
            fw[l] = MA[0]

    fwmapped = fw[labels]

    return fwmapped


def forward_map(fw, labels, labelsets, delete_labelsets=False):
    """Map all labels in value to key."""

    for lsk, lsv in labelsets.items():
        lsv = sorted(list(lsv))
        for l in lsv:
            if delete_labelsets:
                fw[l] = 0
            else:
                fw[l] = lsk

    fw[0] = 0
    fwmapped = fw[labels]

    return fwmapped


def save_step(outpaths, dsname, ds, elsize, axlab):
    """Save intermediate result as a dataset in the outputfile."""

    try:
        h5path = outpaths[dsname]
    except KeyError:
        pass
    else:
        h5_write(ds, ds.shape, ds.dtype,
                 h5path,
                 element_size_um=elsize, axislabels=axlab)


def read_labelsets(lsfile):
    """Read labelsets from file."""

    e = os.path.splitext(lsfile)[1]
    if e == '.pickle':
        with open(lsfile, 'rb') as f:
            labelsets = pickle.load(f)
    else:
        labelsets = read_labelsets_from_txt(lsfile)

    return labelsets


def read_labelsets_from_txt(lsfile):
    """Read labelsets from a textfile."""

    labelsets = {}

    with open(lsfile) as f:
        lines = f.readlines()
        for line in lines:
            splitline = line.split(':', 2)
            lsk = int(splitline[0])
            lsv = set(np.fromstring(splitline[1], dtype=int, sep=' '))
            labelsets[lsk] = lsv

    return labelsets


def write_labelsets(labelsets, filestem, filetypes='txt'):
    """Write labelsets to file."""

    if 'txt' in filetypes:
        filepath = filestem + '.txt'
        write_labelsets_to_txt(labelsets, filepath)
    if 'pickle' in filetypes:
        filepath = filestem + '.pickle'
        with open(filepath, "wb") as f:
            pickle.dump(labelsets, f)


def write_labelsets_to_txt(labelsets, filepath):
    """Write labelsets to a textfile."""

    with open(filepath, "w") as f:
        for lsk, lsv in labelsets.items():
            f.write("%8d: " % lsk)
            ls = sorted(list(lsv))
            for l in ls:
                f.write("%8d " % l)
            f.write('\n')


def filter_on_size(labels, labelset, min_labelsize, remove_small_labels=False,
                   save_steps=True, root='', ds_name='',
                   outpaths=[], element_size_um=None, axislabels=None):
    """Filter small labels from a volume; write the set to file."""

    if not min_labelsize:
        return labels, set([]), set([])

    areas = np.bincount(labels.ravel())
    fwmask = areas < min_labelsize
    ls_small = set([l for sl in np.argwhere(fwmask) for l in sl])
    ls_small &= labelset
    print('number of small labels: {}'.format(len(ls_small)))

    smalllabelmask = np.array(fwmask, dtype='bool')[labels]

    if save_steps:
        filestem = '{}_{}_smalllabels'.format(root, ds_name)
        write_labelsets({0: ls_small}, filestem, ['txt', 'pickle'])
        save_step(outpaths, 'smalllabelmask', smalllabelmask,
                  element_size_um, axislabels)

    if remove_small_labels:
        labelset -= ls_small

        labels[smalllabelmask] = 0

        if save_steps:
            filestem = '{}_{}_largelabels'.format(root, ds_name)
            write_labelsets({0: labelset}, filestem, ['txt', 'pickle'])
            save_step(outpaths, 'largelabels', labels,
                      element_size_um, axislabels)

    return labels, labelset, ls_small


def load_config(filepath, run_from='', run_upto='', run_only=''):
    """Load a configuration for 2D myelin segmentation."""

    configdir, configfile = os.path.split(os.path.realpath(filepath))
    if configdir not in sys.path:
        sys.path.append(configdir)
    conf = importlib.import_module(os.path.splitext(configfile)[0])

    return conf.config(run_from=run_from, run_upto=run_upto, run_only=run_only)


def normalize_datasets(datadir, datasets, postfix='_norm.tif'):
    """Normalize the datasets on their global signal."""

    data_ref = imread(os.path.join(datadir, datasets[0] + '.tif'))
    mean_ref = np.sum(data_ref) / data_ref.size

    imsave(os.path.join(datadir, datasets[0] + postfix),
           data_ref)

    for dataset in datasets[1:]:
        data_tar = imread(os.path.join(datadir, dataset + '.tif'))
        mean_tar = np.sum(data_tar) / data_tar.size
        data_tar_norm = data_tar * (mean_ref / mean_tar)
        imsave(os.path.join(datadir, dataset + postfix),
               data_tar_norm.astype(data_tar.dtype))


def nii2h5(niipath, h5path_out,
           inlayout='xyz', outlayout='zyx',
           protective=False):
    """Convert nifti to hdf5."""

    # check output path
    if '.h5' in h5path_out:
        status, info = h5_check(h5path_out, protective)
        print(info)
        if status == "CANCELLED":
            return

    # load nifti data
    data, elsize, axlab, _ = nii_load(
        niipath, load_data=True,
        inlayout=inlayout, outlayout=outlayout
        )

    # open data for writing
    h5file_out, ds_out = h5_write(None, data.shape, data.dtype,
                                  h5path_out,
                                  element_size_um=elsize,
                                  axislabels=axlab)
    ds_out[:] = data
    h5file_out.close()


def h5_del(h5path_full):
    """Delete a hdf5 dataset."""

    basepath, h5path_dset = h5path_full.split('.h5/')
    h5path_file = basepath + '.h5'
    with h5py.File(h5path_file, 'a') as f:
        if f.get(h5path_dset):
            del f[h5path_dset]


def tif2h5_3D(datadir, dataset, dset,
              elsize, axlab, postfix='_norm.tif'):
    """Convert 2D tif to 3D h5."""

    h5path = os.path.join(datadir, dataset + '.h5')
    h5file = h5py.File(h5path, 'a')

    tifpath = os.path.join(datadir, dataset + postfix)
    data = imread(tifpath, as_grey=True)

    dsshape = (1,) + tuple(data.shape)
    ds0 = h5file.create_dataset(dset, shape=dsshape, dtype=data.dtype)
    ds0[0, :, :] = data

    ds0.attrs['element_size_um'] = elsize
    for i, label in enumerate(axlab):
        ds0.dims[i].label = label

    h5file.close()


def tif2h5_4D(datadir, dataset, dset,
              elsize, axlab, postfix='.tif'):
    """Convert 3D tif to 4D h5."""

    h5path = os.path.join(datadir, dataset + '.h5')
    h5file = h5py.File(h5path, 'a')

    tifpath = os.path.join(datadir, dataset + postfix)
    data = imread(tifpath)

    dsshape = (1,) + tuple(data.shape)
    ds0 = h5file.create_dataset(dset, shape=dsshape, dtype=data.dtype)
    ds0[0, :, :, :] = data

    ds0.attrs['element_size_um'] = elsize
    for i, label in enumerate(axlab):
        ds0.dims[i].label = label

    h5file.close()


def split_and_permute(h5path, dset0, dsets):
    """Split and transpose h5 dataset.

    This corrects for matlab's wonky h5 output after EED filter.
    """

    h5file = h5py.File(h5path, 'a')
    ds0 = h5file[dset0]
    data = ds0[:].transpose()

    dsshape = tuple(data.shape[:3])
    for i, dset in enumerate(dsets):
        ds = h5file.create_dataset(dset, shape=dsshape, dtype=data.dtype)
        ds[0, :, :] = data[0, :, :, i]

    h5file.close()


def copy_attributes(h5path, dset0, dsets):
    """Copy attributes from one h5 dataset to another."""

    h5file = h5py.File(h5path, 'a')
    ds0 = h5file[dset0]

    for dset in dsets:
        ds = h5file[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label

    h5file.close()


def shuffle_labels(labels):
    """Shuffle labels to a random order."""

    ulabels = np.unique(labels)
    maxlabel = np.amax(ulabels)
    fw = np.array([l if l in ulabels else 0
                   for l in range(0, maxlabel + 1)])
    mask = fw != 0
    fw_nz = fw[mask]
    shuffle(fw_nz)
    fw[mask] = fw_nz

    return fw


def gen_steps(outpaths, save_steps=True):
    """Generate paths for steps output."""

    fileformat = get_format(outpaths['out'])

    if fileformat == '.h5':
        ext = '.h5'
        root, ds_main = outpaths['out'].split(ext)  # ds_main includes filesep
    else:
        if fileformat == '.nii':
            ext = '.nii.gz'
            root, _ = outpaths['out'].split(ext)
        elif fileformat == '.tif':
            ext = '.tif'
            root, _ = outpaths['out'].split(ext)
        else:  # FIXME: generalize
            ext = ''
            root = outpaths['out']
        comps = root.split('_')
        root = '_'.join(comps[:-1])
        ds_main = comps[-1]

    groupname = ds_main + "_steps"

    for dsname, outpath in outpaths.items():
        if (dsname == 'out'):
            continue
        if save_steps:
            if not outpath:
                outpaths[dsname] = get_outpath(root, groupname, dsname, ext)
        else:
            del outpaths[dsname]

    return outpaths


def get_outpath(root, groupname, dsname, ext='.h5'):
    """Generate paths for steps output."""

    if ext == '.h5':
        outpath = os.path.join(root + ext + groupname, dsname)
    else:
        outpath = root + '_' + groupname + '_' + dsname + ext

    return outpath


def get_format(filepath):

    fileformat = '.tifs'
    for ext in ['.h5', '.nii', '.tif']:
        if ext in filepath:
            fileformat = ext
            return fileformat


def get_blocks(im, blocksize, margin=[], blockrange=[], path_tpl=''):
    """Create a list of dictionaries with data block info.

    TODO: step?
    """

    shape = list(slices2sizes(im.slices))

    if not blocksize:
        blocksize = [dim for dim in shape]
    if not margin:
        margin = [0 for dim in shape]

    blocksize = [dim if bs == 0 else bs for bs, dim in zip(blocksize, shape)]

    starts, stops, blocks = {}, {}, []
    for i, dim in enumerate(im.axlab):
        starts[dim], stops[dim] = get_blockbounds(im.slices[i].start,
                                                  shape[i],
                                                  blocksize[i],
                                                  margin[i])

    ndim = len(im.axlab)
    starts = tuple(starts[dim] for dim in im.axlab)
    stops = tuple(stops[dim] for dim in im.axlab)
    startsgrid = np.array(np.meshgrid(*starts))
    stopsgrid = np.array(np.meshgrid(*stops))
    starts = np.transpose(np.reshape(startsgrid, [ndim, -1]))
    stops = np.transpose(np.reshape(stopsgrid, [ndim, -1]))

    idstring = '{:05d}-{:05d}_{:05d}-{:05d}_{:05d}-{:05d}'
    for start, stop in zip(starts, stops):

        block = {}
        block['slices'] = [slice(sta, sto) for sta, sto in zip(start, stop)]

        x = block['slices'][im.axlab.index('x')]
        y = block['slices'][im.axlab.index('y')]
        z = block['slices'][im.axlab.index('z')]
        block['id'] = idstring.format(x.start, x.stop,
                                      y.start, y.stop,
                                      z.start, z.stop)
#         block['dataslices'] = slices2dataslices(block['slices'])
#         block['size'] = list(im.slices2shape(slices=block['slices']))
        block['path'] = path_tpl.format(block['id'])

        blocks.append(block)

    if blockrange:
        blocks = blocks[blockrange[0]:blockrange[1]]

    return blocks


def slices2dataslices(slcs):

    dataslices = []
    for slc in slcs:
        dataslices += [slc.start, slc.stop, slc.step]

    return dataslices


def get_blockbounds(offset, shape, blocksize, margin):
    """Get the block range for a dimension."""

    # blocks
    starts = range(offset, shape + offset, blocksize)
    stops = np.array(starts) + blocksize

    # blocks with margin
    starts = np.array(starts) - margin
    stops = np.array(stops) + margin

    # blocks with margin reduced on boundary blocks
    starts[starts < offset] = offset
    stops[stops > shape + offset] = shape + offset

    return starts, stops


def get_image(image_in, imtype='', **kwargs):

    comm = kwargs.pop('comm', None)
    load_data = kwargs.pop('load_data', True)

    if isinstance(image_in, Image):
        im = image_in
        if 'slices' in kwargs.keys():
            im.slices = kwargs['slices']
        if im.format == '.h5':
            im.h5_load(comm, load_data)
    else:
        if imtype == 'Label':
            im = LabelImage(image_in, **kwargs)
        elif imtype == 'Mask':
            im = MaskImage(image_in, **kwargs)
        else:
            im = Image(image_in, **kwargs)

        im.load(comm, load_data)

    return im


def dump_labelsets(labelsets, comps, rank):
    """Dump labelsets from a single process in a pickle."""

    mname = "host-{}_rank-{:02d}".format(socket.gethostname(), rank)
    lsroot = '{}_{}_{}'.format(comps['base'], comps['dset'], mname)
    write_labelsets(labelsets, lsroot, ['pickle'])


def combine_labelsets(labelsets, comps):
    """Combine labelsets from separate processes."""

    lsroot = '{}_{}'.format(comps['base'], comps['dset'])
    match = "{}_host*_rank*.pickle".format(lsroot)
    infiles = glob.glob(match)
    for ppath in infiles:
        with open(ppath, "rb") as f:
            newlabelsets = pickle.load(f)
        for lsk, lsv in newlabelsets.items():
            labelsets = classify_label_set(labelsets, lsv, lsk)

    write_labelsets(labelsets, lsroot, ['txt', 'pickle'])

    return labelsets


def transpose(prop, in2out):

    if prop is not None:
        prop = tuple([prop[i] for i in in2out])

    return prop

