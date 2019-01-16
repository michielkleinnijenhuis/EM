import os
import h5py
import glob
import numpy as np
import nibabel as nib
from skimage.io import imread, imsave

import errno

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

try:
    import DM3lib as dm3
except ImportError:
    raise


class Image:
    """

    """

    def __init__(self, path='',
                 elsize=None, axlab=None, dtype='float',
                 shape=None, dataslices=None,
                 chunks=None, compression='gzip',
                 protective=False):

        self.path = path
        self.elsize = elsize
        self.axlab = axlab
        self.dtype = dtype
        self.dims = shape
        self.dataslices = dataslices
        self.chunks = chunks
        self.compression = compression
        self.protective = protective

        self.file = None
        self.ds = []

        self.set_format()

    def set_format(self):
        """Set the format of the image."""

        if os.path.isdir(self.path):
            files = sorted(glob.glob(os.path.join(self.path, '*')))
            path = files[0]
            if '.tif' in path:
                self.format = '.tifs'
                return
        else:
            path = self.path

        for ext in ['.h5', '.nii', '.dm3', '.tif']:
            if ext in path:
                self.format = ext
                return

#         self.format = '.tifs'
        self.format = '.dat'

    def output_check(self, outpaths, save_steps=True, protective=False):
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
                status = self.output_check_all(root, ext, None,
                                               save_steps, protective)
                if status == "CANCELLED":
                    return status

        # validate the main output
        root, ext = os.path.splitext(outpaths['out'])
        status = self.output_check_all(root, ext, outpaths,
                                       save_steps, protective)

        return status

    def output_check_all(self, root, ext, outpaths=None, save_steps=False, protective=False):

        if '.h5' in ext:
            if outpaths is None:
                status = self.output_check_h5(['{}{}'.format(root, ext)], save_steps, protective)
            else:
                status = self.output_check_h5(outpaths, save_steps, protective)
        elif '.nii' in ext:
            status = self.output_check_dir(['{}{}'.format(root, ext)], protective)
        else:  # directory with images assumed
            status = self.output_check_dir([root], protective)

        return status

    def output_check_h5(self, outpaths, save_steps=True, protective=False):

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
                    status, info = self.h5_check(outpath, protective)
                    print("{}: {}".format(status, info))
                    if status == "CANCELLED":
                        return status

    def output_check_dir(self, outpaths, protective):
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

    def h5_check(self):
        """Check if dataset exists in a h5 file."""

        h5path = self.h5_split()  # TODO: groups
        try:
            h5file = h5py.File(h5path['file'], 'r+')
        except IOError:  # FIXME: it's okay for the file not to be there
            status = "INFO"
            info = "could not open {}".format(self.path)
#             raise UserWarning("could not open {}".format(self.path))
        else:
            if h5path['dset'] in h5file:
                if self.protective:  # TODO: raise error
                    status = "CANCELLED"
                    info = "protecting {}".format(self.path)
                    raise Exception("protecting {}".format(self.path))
                else:
                    status = "WARNING"
                    info = "overwriting {}".format(self.path)
#                     raise UserWarning("overwriting {}".format(self.path))
            else:
                status = "INFO"
                info = "writing to {}".format(self.path)
#                 raise UserWarning("writing to {}".format(self.path))
            h5file.close()
            print(info)

    def h5_check_chunks(self):
        """Make sure chunksize does not exceed dimensions."""

        if self.chunks is None:
            return

        self.chunks = tuple([cs if cs < dim else dim
                             for cs, dim in zip(list(self.chunks), self.dims)])

    def split_path(self, filepath=''):
        """Split path into components."""

        filepath = filepath or self.path

        comps = {}

        if os.path.isdir(filepath):
            comps['dir'] = filepath
            return comps

        if self.format == '.tifs':
            comps['ext'] = 'tif'
        elif self.format == '.nii':
            comps['ext'] = '.nii.gz'    # TODO: non-zipped
        else:
            comps['ext'] = self.format

        if comps['ext'] not in filepath:
            raise Exception('{} not in path'.format(comps['ext']))

        comps['base'] = filepath.split(comps['ext'])[0]
        comps['dir'], comps['fname'] = os.path.split(comps['base'])
        comps['file'] = comps['base'] + comps['ext']

        if comps['ext'] == '.h5':
            comps_int = self.h5_split_int(filepath.split('.h5')[1])
            comps.update(comps_int)
        else:
            pass  # TODO: groups/dset from fname

        return comps

    def h5_split_int(self, path_int=''):
        """Split components of a h5 path."""

        path_int = path_int or self.path.split('.h5')[1]

        comps = {}

        if '/' not in path_int:
            raise Exception('no groups or dataset specified for .h5 path')

        int_comp = path_int.split('/')
        comps['groups'] = int_comp[1:-1]
        comps['dset'] = int_comp[-1]

        return comps

    def h5_split(self, ext='.h5'):
        """Split components of a h5 path."""

        h5path = {}

        h5path['ext'] = ext

        if h5path['ext'] not in self.path:
            raise Exception('{} not in path'.format(h5path['ext']))

        h5path['base'], h5path['int'] = self.path.split('.h5')

        if '/' not in h5path['int']:
            raise Exception('no groups or dataset specified for .h5 path')

        h5path_int_comp = h5path['int'].split('/')
        h5path['groups'] = h5path_int_comp[1:-1]
        h5path['dset'] = h5path_int_comp[-1]

        h5path['file'] = h5path['base'] + h5path['ext']
        h5path['dir'], h5path['fname'] = os.path.split(h5path['base'])

        return h5path

    def nii_split(self):
        """Split components of a nii path."""

        path_comps = {}

        if '.nii' not in self.path:
            raise Exception('.nii not in path')

        path_comps['base'] = self.path.split('.nii')[0]
        path_comps['ext'] = '.nii.gz'  # FIXME: non-zipped
#         path_comps['dir'], path_comps['fname'] = os.path.split(path_comps['base'])

#         h5path_int_comp = h5path['int'].split('/')
#         h5path['groups'] = h5path_int_comp[1:-1]
#         h5path['dset'] = h5path_int_comp[-1]
# 
#         h5path['file'] = h5path['base'] + '.h5'

        return path_comps

    def h5_open(self, permission, comm=None):
        """Open a h5 file."""

        h5path = self.h5_split()
#         if isinstance(self.file, h5py.File):
#             pass
#         else:
        if comm is None:
            self.file = h5py.File(h5path['file'], permission)
        else:
            self.file = h5py.File(h5path['file'], permission,
                                  driver='mpio', comm=comm)

    def load(self, comm=None, load_data=True):
        """Load a dataset."""

        if not self.path:
            pass

        formats = {'.h5': self.h5_load,
                   '.nii': self.nii_load,
                   '.dm3': self.dm3_load,
                   '.tif': self.tif_load,
                   '.tifs': self.tifs_load}

        formats[self.format](comm, load_data)

        self.elsize = self.get_elsize()
        self.axlab = self.get_axlab()

    def h5_load(self, comm=None, load_data=True):
        """Load a h5 dataset."""

        self.h5_open('r+', comm)
        h5path = self.h5_split()
        self.ds = self.file[h5path['int']]
        self.dims = self.ds.shape
        self.dtype = self.ds.dtype
        self.chunks = self.ds.chunks

#         if load_data:
#             data, slices = self.load_dataset()

    def nii_load(self, comm=None, load_data=True):
        """Load a nifti dataset."""

        self.file = nib.load(self.path)
        self.ds = self.file.dataobj
        self.dims = self.ds.shape
        self.dtype = self.ds.dtype

#         if load_data:
#             data, slices = self.load_dataset()

    def tif_load(self, comm=None, load_data=True):
        """Load a 3D tif dataset."""

        self.file = None
        self.ds = imread(self.path)
        self.dims = self.ds.shape
        self.dtype = self.ds.dtype

        if load_data:
            self.ds = imread(self.path)

    def tifs_load(self, comm=None, load_data=True):
        """Load a stack of tifs."""

        self.file = sorted(glob.glob(os.path.join(self.path, "*.tif*")))
        self.ds = []
        self.dims = np.array([len(self.file)] + self.tif_get_yxdims())
        self.dtype = self.tif_load_dtype()

        if load_data:  # TODO: dataslices (see dm3)
            for fpath in self.file:
                self.ds.append(self.get_image(fpath))
            self.ds = np.array(self.ds, dtype=self.dtype)

    def dm3_load(self, comm=None, load_data=True):
        """Load a stack of dm3s."""

        self.file = sorted(glob.glob(os.path.join(self.path, "*.dm3")))
        self.ds = []
        self.dims = np.array([len(self.file)] + self.dm3_get_yxdims())
        self.dtype = self.dm3_load_dtype()

        if load_data:
            slcs = self.get_slice_objects()
            slcs_z = slcs[self.axlab.index('z')]
            for fpath in self.file[slcs_z]:
                im = self.get_image(fpath)
                im = im[slcs[self.axlab.index('y')],
                        slcs[self.axlab.index('x')]]
                self.ds.append(im)
            self.ds = np.array(self.ds, dtype=self.dtype)

    def squeeze_channel(self, dim):

        props = {'dims': self.ds.shape,
                 'es': self.elsize,
                 'al': self.axlab,
                 'chunks': self.chunks}

        if self.get_ndim() != 4:  # FIXME: generalize
            if self.dataslices is not None:
                props['dslcs'] = list(self.dataslices)
            else:
                props['dslcs'] = self.dataslices
            return props

        squeezed = {}
        for prop, val in props.items():
            if val is not None:
                squeezed[prop] = list(val)
                del squeezed[prop][dim]
            else:
                squeezed[prop] = val

        dslcs = self.dataslices
        if dslcs is not None:
            squeezed['dslcs'] = list(dslcs)
            del squeezed['dslcs'][dim*3:dim*3+3]
        else:
            squeezed['dslcs'] = None

        return squeezed

    def slice_dataset(self):

        ndim = self.get_ndim()
        slices = self.get_slice_objects()

        if ndim == 1:
            data = self.ds[slices[0]]
        elif ndim == 2:
            data = self.ds[slices[0],
                           slices[1]]
        elif ndim == 3:
            data = self.ds[slices[0],
                           slices[1],
                           slices[2]]
        elif ndim == 4:
            data = self.ds[slices[0],
                           slices[1],
                           slices[2],
                           slices[3]]
        elif ndim == 5:
            data = self.ds[slices[0],
                           slices[1],
                           slices[2],
                           slices[3],
                           slices[4]]

#         print(data.shape, np.nonzero(data.shape == 1))
#         for singleton in np.nonzero(data.shape == 1):
#             del self.elsize[singleton]
#             del self.axlab[singleton]
#             del self.chunks[singleton]
#             del self.dataslices[singleton]
# 
#         self.dims = len(self.elsize)
#         dslcs = [[0, dim, 1] for dim in self.dims]
#         self.dataslices = [item for dslc in dslcs for item in dslc]

        self.ds = np.squeeze(data)

        return np.squeeze(data)

    def load_dataset(self):

        self.slice_dataset()
        self.format = '.dat'
        self.file = None
        self.set_dims()

    def load_dataset_old(self, outlayout='', uint8conv=False):
        """Load data from a proxy and select/transpose/convert/...."""

        slices = self.get_slice_objects(self.ds.shape)
        data = self.slice_dataset()

        if list(self.axlab) != list(outlayout):
            in2out = [self.axlab.index(l) for l in outlayout]
            data = np.transpose(data, in2out)
            self.elsize = np.array(self.elsize)[in2out]
            self.axlab = outlayout
            slices = [slices[i] for i in in2out]

        if self.dtype:
            data = data.astype(self.dtype, copy=False)

        if uint8conv:
            from skimage import img_as_ubyte
            data = self.normalize_data(data)[0]
            data = img_as_ubyte(data)

        return data, slices

    def normalize_data(self, data):
        """Normalize data between 0 and 1."""

        data = data.astype('float64')
        datamin = np.amin(data)
        datamax = np.amax(data)
        data -= datamin
        data *= 1/(datamax-datamin)

        return data, [datamin, datamax]

    def get_ndim(self):
        """Return the cardinality of the dataset."""

        if self.format == '.h5':
            ndim = self.ds.ndim
        elif self.format == '.nii':
            ndim = len(self.file.header.get_data_shape())
        elif self.format == '.tif':
            ndim = self.ds.ndim
        elif self.format == '.tifs':
            ndim = self.ds.ndim
        elif self.format == '.dm3':
            ndim = len(self.dims)
        elif self.format == '.dat':
            ndim = self.ds.ndim

        return ndim

    def set_dims(self):
        """Set the dims property to the shape of the dataset."""

        self.dims = self.ds.shape

    def set_dataslices(self):
        """Set the dataslices property to the full shape of the dataset."""

        if self.dataslices is None:
            self.dataslices = []
            for dim in self.dims:
                self.dataslices += [0, dim, 1]

    def get_slice_objects(self, dataslices=None, dims=None):
        """Get the full ranges for z, y, x if upper bound is undefined."""

        if dataslices is None:
            self.set_dataslices()
            dataslices = self.dataslices
        if dims is None:
            dims = self.dims

        starts = dataslices[::3]
        stops = dataslices[1::3]
        stops = [dim if stop == 0 else stop
                 for stop, dim in zip(stops, dims)]
        steps = dataslices[2::3]
        slices = [slice(start, stop, step)
                  for start, stop, step in zip(starts, stops, steps)]

        return slices

    def get_slice_objects_prc(self, dataslices, zyxdims):
        """Get the full ranges for z, y, x if upper bound is undefined."""

        # set default dataslice
        if dataslices is None:
            dataslices = [0, 0, 1, 0, 0, 1, 0, 0, 1]

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

    def slices2shape(self, dataslices=None, dims=None):
        """Get the shape of the sliced dataset."""

        # FIXME: generalize to 2D/4D
        slices = self.get_slice_objects(dataslices, dims)
#         shape = (len(range(*slc.indices(slc.stop)))
#                  for slc in slices)
        if len(slices) == 3:
            shape = (len(range(*slices[0].indices(slices[0].stop))),
                     len(range(*slices[1].indices(slices[1].stop))),
                     len(range(*slices[2].indices(slices[2].stop))))
        elif len(slices) == 4:
            shape = (len(range(*slices[0].indices(slices[0].stop))),
                     len(range(*slices[1].indices(slices[1].stop))),
                     len(range(*slices[2].indices(slices[2].stop))),
                     len(range(*slices[3].indices(slices[3].stop))))

        return shape

    def get_elsize(self):
        """Get the element sizes."""

        if ((self.elsize is not None) and (len(self.elsize) > 0)):
            elsize = self.elsize
#             print("""WARNING:
#                   elsize already specified as {}""".format(self.elsize))
            return elsize

        formats = {'.h5': self.h5_load_elsize,
                   '.nii': self.nii_load_elsize,
                   '.dm3': self.dm3_load_elsize,
                   '.tif': self.tif_load_elsize,
                   '.tifs': self.tifs_load_elsize}
        elsize = formats[self.format]()

        if elsize is None:
            elsize = np.array([1.] * self.get_ndim())
#             raise Exception("""WARNING: elsize is None;
#                                replaced by {}""".format(elsize))

        return elsize

    def h5_load_elsize(self):
        """Get the element sizes from a dataset."""

        if 'element_size_um' in self.ds.attrs.keys():
            elsize = self.ds.attrs['element_size_um']
        else:
            elsize = None

        return elsize

    def nii_load_elsize(self):
        """Get the element sizes from a dataset."""

        elsize = list(self.file.header.get_zooms())

        return elsize

    def dm3_load_elsize(self):
        """Get the element sizes from a dataset."""

        dm3f = dm3.DM3(self.file[0], debug=0)

        id_dm3 = 'root.ImageList.1.ImageData'
        tag = '{}.Calibrations.Dimension.{:d}.Scale'
        elsize_x = float(dm3f.tags.get(tag.format(id_dm3, 0)))
        elsize_y = float(dm3f.tags.get(tag.format(id_dm3, 1)))

        tag = 'root.ImageList.1.ImageTags.SBFSEM.Record.Slice thickness'
        slicethickness = float(dm3f.tags.get(tag))
        elsize_z = slicethickness / 1000

        return [elsize_z, elsize_y, elsize_x]

    def tif_load_elsize(self):
        """Get the element sizes from a dataset."""

        elsize = None

        return elsize

    def tifs_load_elsize(self):
        """Get the element sizes from a dataset."""

        elsize = None

        return elsize

    def get_axlab(self):
        """Get the axis labels."""

        if ((self.axlab is not None) and (len(self.axlab) > 0)):
            axlab = ''.join(self.axlab)
#             print("""WARNING:
#                   axlab already specified as {}""".format(self.axlab))
            return axlab

        formats = {'.h5': self.h5_load_axlab,
                   '.nii': self.nii_load_axlab,
                   '.dm3': self.dm3_load_axlab,
                   '.tif': self.tif_load_axlab,
                   '.tifs': self.tifs_load_axlab}
        axlab = formats[self.format]()

        if axlab is None:
            axlab = 'zyxct'[:self.get_ndim()]
#             raise Exception("""WARNING: axlab is None;
#                                replaced by {}""".format(axlab))

        return axlab

    def h5_load_axlab(self):
        """Get the dimension labels from a dataset."""

        if 'DIMENSION_LABELS' in self.ds.attrs.keys():
            axlab = ''.join(self.ds.attrs['DIMENSION_LABELS'])
        else:
            axlab = None

        return axlab

    def nii_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'xyztc'[:self.get_ndim()]  # FIXME: get from header?

        return axlab

    def dm3_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'zyxct'[:self.get_ndim()]  # FIXME: get from header?

        return axlab

    def tif_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'zyxct'[:len(self.get_ndim())]  # FIXME: get from header?

        return axlab

    def tifs_load_axlab(self):
        """Get the dimension labels from a dataset."""

        axlab = 'zyxct'[:len(self.get_ndim())]  # FIXME: get from header?

        return axlab

    def h5_load_attributes(self):
        """Load attributes from a dataset."""

        ndim = self.get_ndim()
        self.h5_load_elsize(np.array([1] * ndim))
        self.h5_load_axlab('zyxct'[:ndim])

    def h5_write_elsize(self):
        """Write the element sizes to a dataset."""

        if self.elsize is not None:
            self.ds.attrs['element_size_um'] = self.elsize

    def h5_write_axlab(self):
        """Write the dimension labels to a dataset."""

        if self.axlab is not None:
            for i, label in enumerate(self.axlab):
                self.ds.dims[i].label = label

    def h5_write_attributes(self):
        """Write attributes to a dataset."""

        self.h5_write_elsize()
        self.h5_write_axlab()

    def create(self, comm=None):
        """Create a dataset."""

#         self.output_check()  # TODO
        if not self.path:
            pass

        formats = {'.h5': self.h5_create,
                   '.nii': self.nii_create,
                   '.tif': self.tif_create,
                   '.tifs': self.tifs_create,
                   '.dat': self.dat_create}

        formats[self.format](comm)

    def h5_create(self, comm=None):
        """Create a h5 dataset."""

        if comm is not None:
            self.chunks = None
            self.compression = None

        self.h5_open('a', comm)

        h5path = self.h5_split()

        parent = self.file
        for grpname in h5path['groups']:
            try:
                parent = parent[grpname]
            except KeyError:
                parent = parent.create_group(grpname)

        if h5path['dset'] in parent:
            self.ds = parent[h5path['dset']]
        else:
            self.h5_check_chunks()
            self.h5_create_dset(parent, h5path['dset'], comm)
            self.h5_write_attributes()

    def h5_create_dset(self, parent, dset_name, comm=None):
        """Create a h5 dataset."""

        if comm is None:
            self.ds = parent.create_dataset(dset_name,
                                            shape=self.dims,
                                            dtype=self.dtype,
                                            chunks=self.chunks,
                                            compression=self.compression)
        else:
            self.ds = parent.create_dataset(dset_name,
                                            shape=self.dims,
                                            dtype=self.dtype)

    def nii_create(self, comm=None):
        """Write a dataset to nifti format."""

        dtype = 'uint8' if self.dtype == 'bool' else self.dtype
        dtype = 'float' if self.dtype == 'float16' else dtype

#         try:
#             self.nii_load()  # FIXME
#         except:  # FileNotFoundError
        self.ds = np.empty(self.dims, dtype)
        self.file = nib.Nifti1Image(self.ds, self.get_transmat())

    def tif_create(self, comm=None):
        """Write a dataset to 3Dtif format."""

#         try:
#             self.tif_load()
#         except:
        self.file = None
        self.ds = np.empty(self.dims, self.dtype)

    def tifs_create(self, comm=None):
        """Write a dataset to a tif stack."""

#         try:
#             self.tifs_load()
#         except:
        self.file = None
        self.ds = np.empty(self.dims, self.dtype)

    def dat_create(self, comm=None):
        """Write a dataset to dat format."""

        self.file = None
        self.ds = np.empty(self.dims, self.dtype)

    def get_transmat(self):
        """Return a transformation matrix with scaling of elsize."""

        mat = np.eye(4)
        if self.elsize is not None:
            mat[0][0] = self.elsize[0]
            mat[1][1] = self.elsize[1]
            mat[2][2] = self.elsize[2]

        return mat

    def write(self, data=None, slices=None):
        """Write a dataset."""

        if data is None:
            data = self.ds

        if slices is None:
            slices = self.get_slice_objects()

        formats = {'.h5': self.h5_write,
                   '.nii': self.nii_write,
                   '.tif': self.tif_write,
                   '.tifs': self.tifs_write,
                   '.dat': self.dat_write}

        formats[self.format](data, slices)

    def h5_write(self, data, slices):
        """Write data to a hdf5 dataset."""

        self.write_block(self.ds, data, slices)

    def nii_write(self, data, slices):
        """Write data to a nifti dataset."""

        if data.dtype == 'bool':
            data = data.astype('uint8')

#         self.ds = self.file.get_fdata()  # loads as floats
#         self.ds = np.asanyarray(self.file.dataobj).astype(self.dtype)  # unsafe cast
        self.ds = np.asanyarray(self.file.dataobj)
        self.write_block(self.ds, data, slices)
        nib.Nifti1Image(self.ds, self.get_transmat()).to_filename(self.path)

    def tif_write(self, data, slices):
        """Write data to a 3Dtif dataset."""

        if data is not None:
            self.write_block(self.ds, data, slices)

        if self.ds.dtype == 'bool':
            dtype = 'uint8'
        else:
            dtype = self.dtype

        imsave(self.path, self.ds.astype(dtype, copy=False))

    def tifs_write(self, data, slices):
        """Write data to a tif stack."""

        # TODO: 4D data?

        nzfills = 5
        ext = '.tif'
        slicedim = self.axlab.index('z')
        slcoffset = slices[slicedim].start

        self.mkdir_p()
        fstring = '{{:0{0}d}}'.format(nzfills)

        if data.dtype == 'bool':
            data = data.astype('uint8')

        if ext != '.tif':
            data = self.normalize_data(data)[0]

        if data.ndim == 2:
            slcno = slcoffset
            filepath = os.path.join(self.path, fstring.format(slcno) + ext)
            imsave(filepath, data)
            return

        for slc in range(0, data.shape[slicedim]):  # TODO: slices
            slcno = slc + slcoffset
            if slicedim == 0:
                slcdata = data[slc, :, :]
            elif slicedim == 1:
                slcdata = data[:, slc, :]
            elif slicedim == 2:
                slcdata = data[:, :, slc]

            filepath = os.path.join(self.path, fstring.format(slcno) + ext)
            imsave(filepath, slcdata)

    def dat_write(self, data, slices):
        """Write data to a dataset."""

        self.write_block(self.ds, data, slices)

    def write_block(self, ds, data, slices):
        """Write a block of data into a dataset."""

        if slices is None:
            slices = self.get_slice_objects()

        ndim = self.get_ndim()

        if ndim == 1:
            ds[slices[0]] = data
        elif ndim == 2:
            ds[slices[0], slices[1]] = data
        elif ndim == 3:
            ds[slices[0], slices[1], slices[2]] = data
        elif ndim == 4:
            ds[slices[0], slices[1], slices[2], slices[3]] = data
        elif ndim == 5:
            ds[slices[0], slices[1], slices[2], slices[3], slices[4]] = data

        return ds

    def close(self):
        """Close a file."""

        if isinstance(self.file, h5py.File):
            self.file.close()

    def get_metadata(self, files, datatype, outlayout, elsize):
        """Get metadata from a dm3 file."""

        # derive the stop-values from the image data if not specified
        if files[0].endswith('.dm3'):
            yxdims, alt_dtype, elsize = self.dm3_get_metadata(files, outlayout, elsize)
        else:
            yxdims = imread(files[0]).shape
            alt_dtype = imread(files[0]).dtype

        zyxdims = [len(files)] + list(yxdims)

        datatype = datatype or alt_dtype

        element_size_um = [0 if el is None else el for el in elsize]

        return zyxdims, datatype, element_size_um

    def dm3_get_metadata(self, files, outlayout, elsize):
        """Get metadata from a dm3 file."""

        try:
            import DM3lib as dm3
        except ImportError:
            raise

        dm3f = dm3.DM3(files[0], debug=0)

#         yxdims = dm3f.imagedata.shape
        alt_dtype = dm3f.imagedata.dtype
#         yxelsize = dm3f.pxsize[0]

        id = 'root.ImageList.1.ImageData'
        tag = '{}.Dimensions.{:d}'
        yxdims = []
        for dim in [0, 1]:
            yxdims += [int(dm3f.tags.get(tag.format(id, dim)))]

        tag = '{}.Calibrations.Dimension.{:d}.Scale'
        for lab, dim in zip('xy', [0, 1]):
            if elsize[outlayout.index(lab)] == -1:
                pxsize = float(dm3f.tags.get(tag.format(id, dim)))
                elsize[outlayout.index(lab)] = pxsize
        tag = 'root.ImageList.1.ImageTags.SBFSEM.Record.Slice thickness'
        slicethickness = float(dm3f.tags.get(tag.format(id, dim)))
        elsize[outlayout.index('z')] = slicethickness / 1000

        return yxdims, alt_dtype, elsize

    def dm3_get_yxdims(self):
        """Get the element sizes from a dm3 dataset."""

        dm3f = dm3.DM3(self.file[0], debug=0)

        id = 'root.ImageList.1.ImageData'
        tag = '{}.Dimensions.{:d}'
        yxdims = []
        for dim in [0, 1]:
            yxdims += [int(dm3f.tags.get(tag.format(id, dim)))]

        return yxdims

    def tif_get_yxdims(self):
        """Get the dimensions of a tif image."""

        if self.file:
            im = imread(self.file[0])
        else:
            im = imread(self.path)

        return list(im.shape)

    def tif_load_dtype(self):
        """Get the datatype of a tif image."""

        if self.file:
            im = imread(self.file[0])
        else:
            im = imread(self.path)

        return im.dtype

    def dm3_load_dtype(self):
        """Get the datatype from a dm3 file."""

        dataTypes = {
            0:  'NULL_DATA',
            1:  'SIGNED_INT16_DATA',
            2:  'REAL4_DATA',
            3:  'COMPLEX8_DATA',
            4:  'OBSELETE_DATA',
            5:  'PACKED_DATA',
            6:  'UNSIGNED_INT8_DATA',
            7:  'SIGNED_INT32_DATA',
            8:  'RGB_DATA',
            9:  'SIGNED_INT8_DATA',
            10: 'UNSIGNED_INT16_DATA',
            11: 'UNSIGNED_INT32_DATA',
            12: 'REAL8_DATA',
            13: 'COMPLEX16_DATA',
            14: 'BINARY_DATA',
            15: 'RGB_UINT8_0_DATA',
            16: 'RGB_UINT8_1_DATA',
            17: 'RGB_UINT16_DATA',
            18: 'RGB_FLOAT32_DATA',
            19: 'RGB_FLOAT64_DATA',
            20: 'RGBA_UINT8_0_DATA',
            21: 'RGBA_UINT8_1_DATA',
            22: 'RGBA_UINT8_2_DATA',
            23: 'RGBA_UINT8_3_DATA',
            24: 'RGBA_UINT16_DATA',
            25: 'RGBA_FLOAT32_DATA',
            26: 'RGBA_FLOAT64_DATA',
            27: 'POINT2_SINT16_0_DATA',
            28: 'POINT2_SINT16_1_DATA',
            29: 'POINT2_SINT32_0_DATA',
            30: 'POINT2_FLOAT32_0_DATA',
            31: 'RECT_SINT16_1_DATA',
            32: 'RECT_SINT32_1_DATA',
            33: 'RECT_FLOAT32_1_DATA',
            34: 'RECT_FLOAT32_0_DATA',
            35: 'SIGNED_INT64_DATA',
            36: 'UNSIGNED_INT64_DATA',
            37: 'LAST_DATA',
            }

        # PIL "raw" decoder modes for the various image dataTypes
        dataTypesDec = {
            1:  'F;16S',     #16-bit LE signed integer
            2:  'F;32F',     #32-bit LE floating point
            6:  'F;8',       #8-bit unsigned integer
            7:  'F;32S',     #32-bit LE signed integer
            9:  'F;8S',      #8-bit signed integer
            10: 'F;16',      #16-bit LE unsigned integer
            11: 'F;32',      #32-bit LE unsigned integer
            14: 'F;8',       #binary
            }

        dtypes = {
            1:  'int16',     #16-bit LE signed integer
            2:  'float32',   #32-bit LE floating point
            6:  'uint8',     #8-bit unsigned integer
            7:  'int32',     #32-bit LE signed integer
            9:  'int8',      #8-bit signed integer
            10: 'uint16',    #16-bit LE unsigned integer
            11: 'uint32',    #32-bit LE unsigned integer
            }

        dm3f = dm3.DM3(self.file[0], debug=0)

        id = 'root.ImageList.1.ImageData'
        tag = '{}.DataType'
        datatype = dtypes[int(dm3f.tags.get(tag.format(id)))]

        return datatype

#     def get_image(self, index):
# 
#         fpath = self.file[index]
# 
#         if fpath.endswith('.dm3'):
#             dm3f = dm3.DM3(fpath, debug=0)
#             im = dm3f.imagedata
#         else:
#             im = imread(fpath)
# 
#         return im

    def get_image(self, fpath):
        """Load dm3 or tif image data."""

        if fpath.endswith('.dm3'):
            dm3f = dm3.DM3(fpath, debug=0)
            im = dm3f.imagedata
        else:
            im = imread(fpath)

        return im

    def transpose(self, outlayout):
        """Transpose the image."""

        in2out = [self.axlab.index(l) for l in outlayout]

        self.axlab = outlayout
        self.elsize = np.array(self.elsize)[in2out]
        self.ds[:] = np.transpose(self.ds[:], in2out)  # FIXME: fail for proxies

        slices = self.get_slice_objects()
        slices = [slices[i] for i in in2out]
        self.dataslices = [item for sl in slices for item in sl]

    def squeeze(self):
        pass

    def mkdir_p(self):
        try:
            os.makedirs(self.path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(self.path):
                pass
            else:
                raise


class MaskImage(Image):

    def invert(self):

        self.ds[:] = ~self.ds[:]


class LabelImage(Image):
    pass
