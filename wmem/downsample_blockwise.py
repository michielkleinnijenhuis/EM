#!/usr/bin/env python

"""Downsample volume by blockwise reduction.

"""

import sys
import argparse

import numpy as np
from skimage.util import view_as_blocks
from scipy.stats import mode as scipy_mode

from wmem import parse, utils, Image


def main(argv):
    """Downsample volume by blockwise reduction."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_downsample_blockwise(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    downsample_blockwise(
        args.inputpath,
        args.dataslices,
        args.blockreduce,
        args.func,
        args.outputpath,
        args.save_steps,
        args.protective,
        )


def downsample_blockwise(
        image_in,
        dataslices=None,
        blockreduce=[3, 3, 3],
        func='np.amax',
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """Downsample volume by blockwise reduction."""

    # Open the inputfile for reading.
    im = utils.get_image(image_in, dataslices=dataslices)

    # Get the matrix size and resolution of the outputdata.
    slicedshape = list(im.slices2shape())
    outsize, elsize = get_new_sizes(func, blockreduce, slicedshape, im.elsize)

    # Open the outputfile for writing and create the dataset or output array.
    mo = Image(outputpath,
               elsize=elsize,
               axlab=im.axlab,
               shape=outsize,
               dtype=im.dtype,
               protective=protective)
    mo.create()

    # Reformat the data to the outputsize.
    if func == 'expand':
        out = im.ds[im.slices[0], im.slices[1], im.slices[2]]  # FIXME: 4D
        for axis in range(0, mo.get_ndim()):
            out = np.repeat(out, blockreduce[axis], axis=axis)
        mo.ds[mo.slices[0], ...] = out
    else:
        """ TODO: flexible mapping from in to out
        now:
        the reduction factor of the first axis must be 1;
        the extent of the remaining axes must be full
        """
        mo.ds[mo.slices[0], ...] = block_reduce(im.ds[im.slices[0], ...],
                                                block_size=tuple(blockreduce),
                                                func=eval(func))

    mo.write()

    im.close()
    mo.close()

    return mo


def get_new_sizes(func, blockreduce, dssize, elsize):
    """Calculate the reduced dataset size and voxelsize."""

    if func == 'expand':
        fun_dssize = lambda d, b: int(np.ceil(float(d) * b))
        fun_elsize = lambda e, b: float(e) / b
    else:
        fun_dssize = lambda d, b: int(np.ceil(float(d) / b))
        fun_elsize = lambda e, b: float(e) * b

    dssize = [fun_dssize(d, b) for d, b in zip(dssize, blockreduce)]
    elsize = [fun_elsize(e, b) for e, b in zip(elsize, blockreduce)]

    return dssize, elsize


# NOTE: adapted version of scikit-image-dev0.13 block_reduce
# it uses flattened blocks to calculate the (scipy) mode
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
        # TODO: implement restriding here instead of reshape?
        outshape = tuple(out.shape[:3]) + tuple([-1])
        out = np.reshape(out, outshape)
        out = scipy_mode(out)
    else:
        for i in range(len(out.shape) // 2):
            out = func(out, axis=-1)

    return out


def mode(array, axis=None):  # axis argument needed for block_reduce
    """Calculate the blockwise mode."""

    smode = np.zeros_like(array[:, :, :, 0])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                block = array[i, j, k, :].ravel()
                smode[i, j, k] = np.argmax(np.bincount(block))

    return smode


if __name__ == "__main__":
    main(sys.argv[1:])
