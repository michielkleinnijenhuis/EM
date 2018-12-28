import h5py
import numpy as np
import nibabel as nib
from skimage.io import imread, imsave


class Image:

    def __init__(self, path='', elsize=None, axlab=None, dtype='float',
                 protective=False):
        self.path = path
        self.elsize = elsize
        self.axlab = axlab
        self.dtype = dtype
        self.dataslices = None
        self.layout = None
        self.chunks = None
        self.compression = 'gzip'
        self.protective = protective
        self.file = None
        self.set_format()

    def set_format(self):
        """Set the format of the image."""

        try:
            FileNotFoundError
        except NameError:
            FileNotFoundError = IOError

        try:
            _, _ = self.path.split('.h5')
        except ValueError:
            pass
        else:
            self.h5_check()
            self.format = '.h5'
            return

        try:
            nib.load(self.path)
        except nib.filebasedimages.ImageFileError:
            pass
        except FileNotFoundError:  # this is py3
            self.format = '.nii'
            return
        except:
            raise
        else:
            self.format = '.nii'
            return

        try:
            imread(self.path)
        except:
            pass
        else:
            self.format = '.img'
            return

    def h5_check(self):
        """Check if dataset exists in a h5 file."""

        _, pd, pf = self.h5_split()
        try:
            h5file = h5py.File(pf, 'r+')
        except IOError:  # FIXME: it's okay for the file not to be there
            status = "INFO"
            info = "could not open {}".format(self.path)
#             raise UserWarning("could not open {}".format(self.path))
        else:
            if pd in h5file:
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

    def load(self, comm=None):
        """Load a dataset."""

        if not self.path:
            pass

        formats = {'.h5': self.h5_load,
                   '.nii': self.nii_load,
                   '.img': self.img_load}

        formats[self.format](comm)

    def h5_load(self, comm=None):
        """Load a h5 dataset."""

        _, pd, pf = self.h5_split()

        self.h5_open(pf, 'r+', comm)
        self.ds = self.file[pd]  # proxy

        ndim = self.get_ndim()
        self.h5_load_elsize(np.array([1] * ndim))
        self.h5_load_axlab('zyxct'[:ndim])
#         da = {'element_size_um': (self.elsize, np.array([1] * ndim)),
#               'DIMENSION_LABELS': (self.axlab, 'zyxct'[:ndim])}
#         self.h5_load_attributes(da)

    def nii_load(self, comm=None):
        """Load a nifti dataset."""

        self.file = nib.load(self.path)

        self.ds = self.file.dataobj  # proxy

        self.elsize = list(self.file.header.get_zooms())
        ndim = len(self.elsize)
#         self.axlab = 'xyztc'[:ndim]  # FIXME: get from header?
        self.axlab = self.axlab or 'xyztc'[:ndim]

    def img_load(self, comm=None):
        """Load a image."""

        self.file = None

        self.ds = imread(self.path)

        self.elsize = np.array([1, 1])
        self.axlab = 'yxc'

    def load_dataset(self, outlayout='', uint8conv=False):
        """Load data from a proxy and select/transpose/convert/...."""

        slices = self.get_slice_objects(self.ds.shape)

        try:
            ndim = self.ds.ndim
        except AttributeError:
            ndim = len(self.ds.dims)

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

    def h5_split(self):
        """Split components of a h5 path."""

        if '.h5' not in self.path:
            raise Exception('.h5 not in path')

        basepath, h5path_dset = self.path.split('.h5')
        h5path_file = basepath + '.h5'

        return basepath, h5path_dset, h5path_file

    def get_ndim(self):
        """Return the cardinality of the dataset."""

        try:
            ndim = self.ds.ndim
        except AttributeError:
            ndim = len(self.ds.dims)

        return ndim

    def get_slice_objects(self, dims):
        """Get the full ranges for z, y, x if upper bound is undefined."""

        # set default dataslices
        if self.dataslices is None:
            self.dataslices = []
            for dim in dims:
                self.dataslices += [0, dim, 1]

        starts = self.dataslices[::3]
        stops = self.dataslices[1::3]
        stops = [dim if stop == 0 else stop
                 for stop, dim in zip(stops, dims)]
        steps = self.dataslices[2::3]
        slices = [slice(start, stop, step)
                  for start, stop, step in zip(starts, stops, steps)]

        return slices

    def h5_load_elsize(self, alt=None):
        """Get the element sizes from a dataset."""

        if 'element_size_um' in self.ds.attrs.keys():
            self.elsize = self.ds.attrs['element_size_um']

        if self.elsize is None:
            self.elsize = alt
            raise Exception("""WARNING: elsize is None;
                               replaced by {}""".format(alt))

    def h5_load_axlab(self, alt=None):
        """Get the dimension labels from a dataset."""

        if 'DIMENSION_LABELS' in self.ds.attrs.keys():
            self.axlab = self.ds.attrs['DIMENSION_LABELS']

        if self.axlab is None:
            self.axlab = alt
            raise Exception("""WARNING: axlab is None;
                               replaced by {}""".format(alt))

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

    def h5_open(self, path_file, permission, comm=None):
        """Open a h5 file."""

        if isinstance(self.file, h5py.File):
            pass
        else:
            if comm is None:
                self.file = h5py.File(path_file, permission)
            else:
                self.file = h5py.File(path_file, permission,
                                      driver='mpio', comm=comm)

    def create(self, shape, comm=None):
        """Create a dataset."""

        if self.format == '.h5':
            self.h5_create(shape, comm)
        elif self.format == '.nii':
            self.nii_create(shape, comm)
        else:
            self.file = None
            self.ds = np.empty(shape, self.dtype)

    def h5_create(self, shape, comm=None):
        """Create a h5 dataset."""

        if comm is not None:
            self.chunks = None
            self.compression = None

        _, pd, pf = self.h5_split()
        self.h5_open(pf, 'a', comm)

        if pd in self.file:
            self.ds = self.file[pd]
        else:
            self.h5_create_dset(pd, shape, comm)
            self.h5_write_attributes()

    def h5_create_dset(self, path_dset, shape, comm=None):
        """Create a h5 dataset."""

        if comm is None:
            self.ds = self.file.create_dataset(path_dset, shape=shape,
                                               dtype=self.dtype,
                                               chunks=self.chunks,
                                               compression=self.compression)
        else:
            self.ds = self.file.create_dataset(path_dset, shape=shape,
                                               dtype=self.dtype)

    def nii_create(self, shape, comm=None):
        """Write a dataset to nifti format."""

        self.ds = np.empty(shape, self.dtype)
        self.file = nib.Nifti1Image(self.ds, self.get_transmat())

    def get_transmat(self):

        mat = np.eye(4)
        if self.elsize is not None:
            mat[0][0] = self.elsize[0]
            mat[1][1] = self.elsize[1]
            mat[2][2] = self.elsize[2]

        return mat

    def nii_load_elsize(self, alt=None):
        """Get the element sizes from a dataset."""

        self.elsize = list(self.file.header.get_zooms())

#         if self.elsize is None:
#             self.elsize = alt
#             raise Exception("""WARNING: elsize is None;
#                                replaced by {}""".format(alt))

    def close(self):
        """Close a file."""

        if isinstance(self.file, h5py.File):
            self.file.close()

    def transpose(self):
        pass


class MaskImage(Image):

    def invert(self):

        self.ds[:] = ~self.ds[:]


class LabelImage(Image):
    pass
