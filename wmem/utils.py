#!/usr/bin/env python

"""General utility functions for the wmem package.

"""

import os
import errno
import pickle

import numpy as np

from skimage.io import imread, imsave
import nibabel as nib

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

import h5py


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


def xyz_datarange(xyz, files):
    """Get the full ranges for x,y,z if upper bound is undefined."""

    firstimage = imread(files[0])

    X = xyz[1] or firstimage.shape[1]
    Y = xyz[3] or firstimage.shape[0]
    Z = xyz[5] or len(files)

    return xyz[0], X, xyz[2], Y, xyz[4], Z


def get_slice_objects_prc(dataslices, files):
    """Get the full ranges for z, y, x if upper bound is undefined."""

    # set default dataslice
    if dataslices is None:
        dataslices = [0, 0, 1, 0, 0, 1, 0, 0, 1]

    # derive the stop-values from the image data if not specified
    firstimage = imread(files[0])
    dataslices[1] = dataslices[1] or len(files)
    dataslices[4] = dataslices[4] or firstimage.shape[0]
    dataslices[7] = dataslices[7] or firstimage.shape[1]

    # create the slice objects
    if dataslices is not None:
        starts = dataslices[::3]
        stops = dataslices[1::3]
        steps = dataslices[2::3]
        slices = [slice(start, stop, step)
                  for start, stop, step in zip(starts, stops, steps)]

    return slices


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


def get_mpi_info(mpi_dtype=MPI.SIGNED_LONG_LONG):
    """Get an MPI communicator."""

    comm = MPI.COMM_WORLD
    mpi_info = {
        'comm': comm,
        'rank': comm.Get_rank(),
        'size': comm.Get_size(),
        'dtype': mpi_dtype
        }

    return mpi_info


def scatter_series(mpi_info, series):
    """Scatter a series of jobnrs over processes."""

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

    try:
        root, ds_main = outpaths['out'].split('.h5')
        h5file_out = h5py.File(root + '.h5', 'a')
    except ValueError:
        status = "CANCELLED"
        info = "main output is not a valid h5 dataset"
        print("{}: {}".format(status, info))
        return status
    else:

        for dsname, outpath in outpaths.items():
            if ((dsname != 'out') and save_steps and (not outpath)):
                grpname = ds_main + "_steps"
                try:
                    h5file_out[grpname]
                except KeyError:
                    h5file_out.create_group(grpname)
                outpaths[dsname] = os.path.join(root + '.h5' + grpname, dsname)

        h5file_out.close()

        for _, outpath in outpaths.items():
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
                return
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
                h5file.__delitem__(h5path_dset)
        else:
            status = "INFO"
            info = "writing to {}".format(h5path_full)

        h5file.close()

    return status, info


def load(dspath, load_data=False,
         dtype='', channels=None,
         inlayout=None, outlayout=None):
    """Load a dataset."""

    try:
        _, _ = dspath.split('.h5')
    except ValueError:
        pass
    else:
        return h5_load(dspath, load_data,
                       dtype, channels,
                       inlayout, outlayout)

    try:
        nib.load(dspath)
    except nib.filebasedimages.ImageFileError:
        pass
    except:
        raise
    else:
        return nii_load(dspath, load_data,
                        dtype, channels,
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


def nii_load(dspath, load_data=False,
             dtype='', channels=None,
             inlayout=None, outlayout=None):
    """Load a nifti dataset."""

    file_in = nib.load(dspath)

    ds_in = file_in.dataobj  # proxy

    elsize = list(file_in.header.get_zooms())
    axlab = 'xyztc'[:ds_in.ndim]  # FIXME: get from header?
    axlab = inlayout or axlab or 'xyztc'[:ds_in.ndim]

    if load_data:
        outlayout = outlayout or axlab
        data, elsize, axlab = load_dataset(ds_in, elsize, axlab,
                                           outlayout, dtype, channels)

        return data, elsize, axlab

    else:

        return file_in, ds_in, elsize, axlab


def h5_load(dspath, load_data=False,
            dtype='', channels=None,
            inlayout=None, outlayout=None):
    """Load a h5 dataset."""

    basepath, h5path_dset = dspath.split('.h5')
    h5path_file = basepath + '.h5'
    file_in = h5py.File(h5path_file, 'r+')

    ds_in = file_in[h5path_dset]  # proxy

    elsize, axlab = h5_load_attributes(ds_in)
    if elsize is None:
        elsize = [1] * ds_in.ndim
    if inlayout is not None:
        axlab = inlayout
    elif axlab is None:
        axlab = 'zyxct'[:ds_in.ndim]

    if load_data:
        outlayout = outlayout or axlab
        data, elsize, axlab = load_dataset(ds_in, elsize, axlab,
                                           outlayout, dtype, channels)
        file_in.close()

        return data, elsize, axlab

    else:

        return file_in, ds_in, elsize, axlab


def load_dataset(ds, elsize, axlab,
                 outlayout, dtype, channels):
    """Load data from a proxy and select/transpose/convert/...."""

    # TODO: make new groups for h5?
    # TODO: test most memory-efficient solution

    if axlab != outlayout:
        in2out = [axlab.index(l) for l in outlayout]
        data = np.transpose(ds[:], in2out)
        elsize = elsize[in2out]
        axlab = outlayout

    # TODO: load data subset/partial view
    if channels is not None:
        data = ds[:, :, :, channels]

    return data.astype(dtype, copy=False), elsize, axlab


def ds_in2out(in2out, ds, elsize):
    """Transpose dataset and element sizes."""

    elsize = elsize[in2out]
    ds = np.transpose(ds, in2out)

    return ds, elsize


def h5_write(data, shape, dtype,
             h5path_full, h5file=None,
             element_size_um=None, axislabels=None,
             chunks=True, compression="gzip",
             usempi=False, driver=None, comm=None):
    """Write a h5 dataset."""

    if usempi:
        chunks = None
        compression = None
        driver = 'mpio'
        comm = comm

    if h5path_full:

        basepath, h5path_dset = h5path_full.split('.h5')
        h5path_file = basepath + '.h5'

#         if not isinstance(h5file, h5py.File):
        h5file = h5py.File(h5path_file, 'a', driver=driver, comm=comm)

        h5ds = h5file.create_dataset(h5path_dset,
                                     shape=shape,
                                     dtype=dtype,
                                     chunks=chunks,
                                     compression=compression)

        h5_write_attributes(h5ds, element_size_um, axislabels)

        if data is not None:

            h5ds[:] = data
            h5file.close()

        else:

            return h5file, h5ds

    else:

        return None, np.empty(shape, dtype)


def h5_load_attributes(h5ds):
    """Get attributes from a dataset."""

    element_size_um = axislabels = None

    if 'element_size_um' in h5ds.attrs.keys():
        element_size_um = h5ds.attrs['element_size_um']

    if 'DIMENSION_LABELS' in h5ds.attrs.keys():
        axislabels = h5ds.attrs['DIMENSION_LABELS']

    return element_size_um, axislabels


def h5_write_attributes(h5ds, element_size_um=None, axislabels=None):
    """Write attributes to a dataset."""

    if element_size_um is not None:
        h5ds.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, label in enumerate(axislabels):
            h5ds.dims[i].label = label


def string_masks(masks, mask):
    """Work out a mask from a series of operations and masks."""

    if not masks:
        return mask

    aliases = {'NOT': 'np.logical_not',
               'AND': 'np.logical_and',
               'OR': 'np.logical_or',
               'XOR': 'np.logical_xor'}

    for m in masks:

        if m in aliases.keys():
            m = aliases[m]

        try:
            _, _ = m.split('.h5')  # ValueError if not a h5 dataset path
            op = eval(m)  # SyntaxError if it does not evaluate
        except ValueError:
            op = eval(m)
            do = False if m == 'np.logical_not' else True
        except SyntaxError:
            newmask = h5_load(m, load_data=True, dtype='bool')[0]
            do = True if m == 'np.logical_not' else False

        if do:
            if m == 'np.logical_not':
                op(newmask, newmask)
            else:
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

    nib.Nifti1Image(data, mat).to_filename(filepath)


def write_to_img(basepath, data, outlayout, nzfills=5, ext='.png'):
    """Write a 3D dataset to a stack of 2D image files."""

    mkdir_p(basepath)
    fstring = '{{:0{0}d}}'.format(nzfills)

    for slc in range(0, data.shape[outlayout.index('z')]):
        slcno = slc  # + slcoffset
        if outlayout.index('z') == 0:
            slcdata = data[slc, :, :]
        elif outlayout.index('z') == 1:
            slcdata = data[:, slc, :]
        elif outlayout.index('z') == 2:
            slcdata = data[:, :, slc]

        filepath = os.path.join(basepath, fstring.format(slcno) + ext)
        imsave(filepath, slcdata)


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
    for lsk, lsv in labelsets.items():
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


def forward_map0(fw, labels, MAlist):
    """Map all labelsets in MAlist to axons."""

    for MA in MAlist:
        MA = sorted(list(MA))
        for l in MA:
            fw[l] = MA[0]

    fwmapped = fw[labels]

    return fwmapped


def forward_map(fw, labels, labelsets, delete_labelsets=False):
    """Map all labelset in value to key."""

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
            lsk = splitline[0]
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

    with open(filepath, "wb") as f:
        for lsk, lsv in labelsets.items():
            f.write("%8d: " % lsk)
            ls = sorted(list(lsv))
            for l in ls:
                f.write("%8d " % l)
            f.write('\n')


def filter_on_size(labels, min_labelsize, remove_small_labels=False,
                   save_steps=True, root='', ds_name='',
                   outpaths=[], element_size_um=None, axislabels=None):
    """Filter small labels from a volume; write the set to file."""

    if not min_labelsize:
        return labels, set([])

    areas = np.bincount(labels.ravel())
    fwmask = areas < min_labelsize
    ls_small = set([l for sl in np.argwhere(fwmask) for l in sl])
    if save_steps:
        filestem = '{}_{}_smalllabels'.format(root, ds_name)
        write_labelsets({0: ls_small}, filestem,
                        filetypes=['txt', 'pickle'])

    if remove_small_labels:
        smalllabelmask = np.array(fwmask, dtype='bool')[labels]
        labels[smalllabelmask] = 0
        if save_steps:
            save_step(outpaths, 'largelabels', labels,
                      element_size_um, axislabels)

    return labels, ls_small
