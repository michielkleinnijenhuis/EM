#!/usr/bin/env python

"""Convert/select/downscale/transpose/... an hdf5 dataset.

"""

import sys
import argparse

import numpy as np

from wmem import parse, utils, wmeMPI, Image


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

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices)

    if dset_name:
        im.slices = utils.dset_name2slices(dset_name, blockoffset,
                                           axlab=inlayout,
                                           shape=im.dims)

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

    data = im.slice_dataset()
    data = np.transpose(data, in2out)

    # TODO: proper general writing astype
    if uint8conv:
        from skimage import img_as_ubyte
        data = utils.normalize_data(data)[0]
        data = img_as_ubyte(data)

    mo.write(data=data)

    im.close()
    mo.close()

    return mo


if __name__ == "__main__":
    main(sys.argv)
