#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np

# from skimage.measure import block_reduce
from skimage.util import view_as_blocks
from scipy.stats import mode

def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('inputstack', nargs=2,
                        help='...')
    parser.add_argument('-d', '--blockreduce', nargs=3, type=int,
                        default=[1,7,7],
                        help='...')
    parser.add_argument('-f', '--func', default='np.amax',
                        help='...')
    parser.add_argument('-o', '--outpf', default='br',
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    inputstack = args.inputstack
    blockreduce = args.blockreduce
    func = args.func
    outpf = args.outpf

    fname = os.path.join(datadir, dset_name + inputstack[0] + '.h5')
    f = h5py.File(fname, 'r')
    fstack = f[inputstack[1]]
    elsize, al = get_h5_attributes(fstack)

    outsize = [int(np.ceil(d/b))
               for d,b in zip(fstack.shape, blockreduce)]

    gname = os.path.join(datadir, dset_name + outpf + inputstack[0] + '.h5')
    g = h5py.File(gname, 'w')
    outds = g.create_dataset(inputstack[1], outsize,
                             dtype=fstack.dtype,
                             compression="gzip")

    outds[:,:,:] = block_reduce(fstack,
                                block_size=tuple(blockreduce),
                                func=eval(func))

    elsize = [e*b for e, b in zip(elsize, blockreduce)]
    write_h5_attributes(outds, elsize, al)

    f.close()
    g.close()


# ========================================================================== #
# function defs
# ========================================================================== #


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


# adapted version of scikit-image-dev0.13 block_reduce
# it switches to flattened blocks to calculate the (scipy) mode
def block_reduce(image, block_size, func=np.sum, cval=0):
    """Down-sample image by applying function to local blocks.
    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    func : callable
        Function object which is used to calculate the return value for each
        local block. This function must implement an ``axis`` parameter such
        as ``numpy.sum`` or ``numpy.min``.
    cval : float
        Constant padding value if image is not perfectly divisible by the
        block size.
    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
    Examples
    --------
    >>> from skimage.measure import block_reduce
    >>> image = np.arange(3*3*4).reshape(3, 3, 4)
    >>> image # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]],
           [[24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35]]])
    >>> block_reduce(image, block_size=(3, 3, 1), func=np.mean)
    array([[[ 16.,  17.,  18.,  19.]]])
    >>> image_max1 = block_reduce(image, block_size=(1, 3, 4), func=np.max)
    >>> image_max1 # doctest: +NORMALIZE_WHITESPACE
    array([[[11]],
           [[23]],
           [[35]]])
    >>> image_max2 = block_reduce(image, block_size=(3, 1, 4), func=np.max)
    >>> image_max2 # doctest: +NORMALIZE_WHITESPACE
    array([[[27],
            [31],
            [35]]])
    """

    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode='constant',
                   constant_values=cval)

    out = view_as_blocks(image, block_size)

    if func is mode:
        ndims = len(out.shape) // 2
        out = np.reshape(out, list(out.shape[:ndims]) + [-1])
        out = np.squeeze(func(out, axis=-1)[0], axis=-1)
    else:
        for i in range(len(out.shape) // 2):
            out = func(out, axis=-1)

    return out


if __name__ == "__main__":
    main(sys.argv[1:])
