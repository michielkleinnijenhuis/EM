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

from wmem import parse, utils, wmeMPI, Image


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
        args.blocksize,
        args.blockmargin,
        args.blockrange,
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
        blocksize=[],
        blockmargin=[],
        blockrange=[],
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

    mpi = wmeMPI(usempi)

    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices,
                         axlab=inlayout, load_data=False)

    # Determine the properties of the output dataset.
    props = {'dtype': datatype or im.dtype,
             'axlab': outlayout or im.axlab,
             'chunks': chunksize or im.chunks,
             'elsize': elsize or im.elsize,
             'shape': list(im.slices2shape())}
    in2out = [im.axlab.index(l) for l in props['axlab']]
    for prop in ['elsize', 'shape', 'chunks']:
        props[prop] = utils.transpose(props[prop], in2out)

    # Open the outputfile for writing and create the dataset or output array.
    mo = Image(outputpath, protective=protective, **props)
    mo.create(comm=mpi.comm)

    in2out_offset = -np.array([slc.start for slc in im.slices])

    # Prepare for processing with MPI.
    """NOTE: using blocked processing is not beneficial here:
       reading 2D slices multiple times.
    """
    if not blocksize:
        blocksize = [0] * im.get_ndim()
        if mo.chunks is not None:
            z_idx = mo.axlab.index('z')
            blocksize[z_idx] = mo.chunks[z_idx]
    mpi.set_blocks(im, blocksize, blockmargin, blockrange)
    mpi.scatter_series()

    # Write blocks of 2D images to the outputfile(s).
    for i in mpi.series:
        block = mpi.blocks[i]

        im.slices = block['slices']
        im.load()

        slcs_out = im.get_offset_slices(in2out_offset)
        slcs_out = utils.transpose(slcs_out, in2out)
        mo.write(data=np.transpose(im.ds, in2out), slices=slcs_out)

    im.close()
    mo.close()

    return mo


if __name__ == "__main__":
    main(sys.argv)
