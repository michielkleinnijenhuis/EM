#!/usr/bin/env python

"""Convert/select/downscale/transpose/... an hdf5 dataset.

"""

import sys
import argparse

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

from wmem import parse, utils, Image


def main(argv):
    """Convert/select/downscale/transpose/... an hdf5 dataset."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_stack2stack(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    stack2stack(
        args.inputpath,
        args.dataslices,
        args.dset_name,
        args.blockoffset,
        args.datatype,
        args.uint8conv,
        args.inlayout,
        args.outlayout,
        args.element_size_um,
        args.chunksize,
        args.outputpath,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def stack2stack(
        image_in,
        dataslices=None,
        dset_name='',
        blockoffset=[],
        datatype=None,
        uint8conv=False,
        inlayout=None,
        outlayout=None,
        elsize=[],
        chunksize=[],
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Convert/select/downscale/transpose/... an hdf5 dataset."""

    mpi_info = utils.get_mpi_info(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, dataslices=dataslices)

    if dset_name:
        im.dataslices = utils.dataset_name2dataslices(dset_name, blockoffset,
                                                      axlab=inlayout,
                                                      shape=im.dims)

    # Determine the properties of the output dataset.
    datatype = datatype or im.dtype
    outlayout = outlayout or im.axlab
    chunksize = chunksize or im.chunks
    elsize = elsize or im.elsize
    outshape = im.slices2shape()

    in2out = [im.axlab.index(l) for l in outlayout]
    if chunksize is not None:
        chunksize = tuple([chunksize[i] for i in in2out])
    elsize = [im.elsize[i] for i in in2out]
    outshape = [outshape[i] for i in in2out]

    # Open the outputfile for writing and create the dataset or output array.
    mo = Image(outputpath,
               elsize=elsize,
               axlab=outlayout,
               shape=outshape,
               chunks=chunksize,
               dtype=datatype,
               protective=protective)
    mo.create(comm=mpi_info['comm'])

    # Prepare for processing with MPI.
    blocksize = [dim for dim in im.dims]
    if mo.chunks is not None:
        blocksize[im.axlab.index('z')] = mo.chunks[mo.axlab.index('z')]  #TODO
    blocks = utils.get_blocks(im.dims, blocksize, [0, 0, 0], im.dataslices)
    series = utils.scatter_series(mpi_info, len(blocks))[0]

    data = im.slice_dataset()
    data = np.transpose(data, in2out)
#     mo.transpose(in2out)

    # TODO: proper general writing astype
    if uint8conv:
        from skimage import img_as_ubyte
        data = utils.normalize_data(data)[0]
        data = img_as_ubyte(data)

    mo.write(data=data)

    im.close()
    mo.close()

    return mo


def get_layouts(im, inlayout, outlayout):

    inlayout = inlayout or ''.join(im.axlab) or 'zyxct'[0:im.get_ndim()]
    outlayout = outlayout or inlayout
    in2out = [inlayout.index(l) for l in outlayout]

    return inlayout, outlayout, in2out


def get_shape(im, in2out, slices):

    outshape = (len(range(*slices[0].indices(slices[0].stop))),
                len(range(*slices[1].indices(slices[1].stop))),
                len(range(*slices[2].indices(slices[2].stop))))
    outshape = [outshape[i] for i in in2out]

    return outshape


def get_chunksize(im, in2out, chunksize):

    if chunksize is not None:
        chunksize = tuple(chunksize) or (
            True if not any(chunksize)
            else (tuple([im.ds.chunks[i] for i in in2out])
                  if im.ds.chunks else None
                  )
            )

    # make sure chunksize does not exceed dimensions
    chunksize = tuple([cs if cs < dim else dim
                       for cs, dim in zip(list(chunksize), im.dims)])

    return chunksize


def get_image(image_in, comm=None, dataslices=None):

    if isinstance(image_in, Image):
        im = image_in
        im.dataslices = dataslices
        if im.format == '.h5':
            im.h5_load()
    else:
        im = Image(image_in, dataslices=dataslices)
        im.load(comm=comm)

    return im


if __name__ == "__main__":
    main(sys.argv)
