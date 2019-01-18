#!/usr/bin/env python

"""Convert a directory of tifs to an hdf5 stack.

Downsampled images are written to the output directory
with the same filename as the input images.
- A subset of the data can be selected with python slicing:
providing <start stop step> in (plane, row, col) order
e.g. setting the flag -D <20 44 1 100 200 1 0 0 1>
selects the images 20 through 43 yielded by the regular expression
and makes a cutout of rows 100 through 199
and with the columns set to the full dimension.
- MPI can be enabled for this function.
- No intermediate results are saved.
- Protective mode will check if images with the same name
already exist in the output directory.
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
    """"Convert a directory of tifs to an hdf5 stack."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_series2stack(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    series2stack(
        args.inputpath,
        args.dataslices,
        args.inlayout,
        args.outlayout,
        args.element_size_um,
        args.datatype,
        args.chunksize,
        args.outputpath,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def series2stack(
        image_in,
        dataslices=None,
        inlayout=None,
        outlayout=None,
        elsize=[],
        datatype=None,
        chunksize=[],
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """"Convert a directory of tifs to an hdf5 stack."""

    mpi_info = utils.get_mpi_info(usempi)

    im = utils.get_image(image_in, comm=mpi_info['comm'],
                         dataslices=dataslices, axlab=inlayout,
                         load_data=False)

    # Determine the properties of the output dataset.
    datatype = datatype or im.dtype
    outlayout = outlayout or im.axlab
    chunksize = chunksize or im.chunks
    elsize = elsize or im.elsize
    outshape = im.slices2shape()

    in2out_offset = -np.array(im.dataslices[::3])
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
    blocksize = [dim for dim in outshape]
    if mo.chunks is not None:
        blocksize[im.axlab.index('z')] = mo.chunks[mo.axlab.index('z')]
    blocks = utils.get_blocks(im, blocksize)
    series = utils.scatter_series(mpi_info, len(blocks))[0]

    # Write blocks of 2D images to the outputfile(s).
    for blocknr in series:
        im.dataslices = blocks[blocknr]['dataslices']
        im.load()

        slcs_out = im.get_slice_objects(im.dataslices, offsets=in2out_offset)
        slcs_out = [slcs_out[i] for i in in2out]
        mo.write(data=np.transpose(im.ds, in2out), slices=slcs_out)

    im.close()
    mo.close()

    return mo


def get_layouts(im, inlayout, outlayout):

    inlayout = inlayout or ''.join(im.axlab) or 'zyxct'[0:im.get_ndim()]
    outlayout = outlayout or inlayout
    in2out = [inlayout.index(l) for l in outlayout]

    return inlayout, outlayout, in2out


def get_shape(im, in2out):

    slices = im.get_slice_objects()
    outshape = (len(range(*slices[0].indices(slices[0].stop))),
                len(range(*slices[1].indices(slices[1].stop))),
                len(range(*slices[2].indices(slices[2].stop))))
    outshape = [outshape[i] for i in in2out]

    return outshape


def get_chunksize(im, in2out, chunksize):

    if chunksize is None:
        return

    if chunksize:
        return tuple(chunksize)

    if not any(chunksize):
        return True

    if im.chunks:
        chunksize = tuple([im.chunks[i] for i in in2out])

    # make sure chunksize does not exceed dimensions
    chunksize = tuple([cs if cs < dim else dim
                       for cs, dim in zip(list(chunksize), im.dims)])

    return chunksize




def get_blocks_dataslices(im, blocksize):

    nblocks = [int(np.ceil(float(dim) / bs))
               for bs, dim in zip(blocksize, im.dims)]

    dslices = []
    for nb in nblocks:
        for blocknr in range(0, nb):
            dataslices = []
            for bs, dim in zip(blocksize, im.dims):
                start = blocknr * bs
                stop = np.minimum(dim, blocknr * bs + bs)
                dataslices += [start, stop, 1]

                dslices.append(dataslices)

    return dslices


def get_blocks_zslices(im, scs):

    dim_z = im.dims[im.axlab.index('z')]
    nblocks = int(np.ceil(float(dim_z) / scs))
    dslices = []
    for blocknr in range(0, nblocks):
        dslices.append(blockedslices(im, scs, blocknr))

    return dslices


def blockedslices(im, scs, blocknr):

    dslices = []
    for al in im.axlab:
        if al == 'z':
            start = blocknr * scs
            zdim = im.dims[im.axlab.index('z')]
            stop = np.minimum(zdim, blocknr * scs + scs)
            dslices += [start, stop, 1]
        else:
            dim = im.dims[im.axlab.index(al)]
            dslices += [0, dim, 1]

    return dslices


def get_blocks_filelist(im, scs):

    # Reshape the file list into a list of blockwise file lists.
    slices = im.get_slice_objects()
    files = im.file[slices[0]]
    files_blocks = zip(* [iter(files)] * scs)
    rem = len(files) % scs
    if rem:
        files_blocks += [tuple(files[-rem:])]


def slices2blockedslices(mo, scs, blocknr):

    # Get slice objects for an output block.
    dslices = []
    for al in mo.axlab:
        if al == 'z':
            dslices += [blocknr * scs, blocknr * scs + scs, 1]
        else:
            dim = mo.dims[mo.axlab.index(al)]
            dslices += [0, dim, 1]
    slices = mo.get_slice_objects(dslices)

    return slices


if __name__ == "__main__":
    main(sys.argv)
