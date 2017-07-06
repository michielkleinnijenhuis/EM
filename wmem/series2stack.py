#!/usr/bin/env python

"""Convert a directory of tifs to an hdf5 stack.

Downsampled images are written to the output directory
with the same filename as the input images.
- A subset of the data can be selected with python slicing:
providing <start stop step> in (plane, row, col) order
e.g. setting the flag -D <20 44 1 100 200 0 0 1>
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
import os
import glob
from math import ceil

import numpy as np
from skimage import io
try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

from wmem import parse, utils


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
        args.inputdir,
        args.regex,
        args.element_size_um,
        args.outlayout,
        args.datatype,
        args.chunksize,
        args.dataslices,
        args.usempi & ('mpi4py' in sys.modules),
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def series2stack(
        inputdir,
        regex='*.tif',
        element_size_um=[1, 1, 1],
        outlayout='zyx',
        datatype='',
        chunksize=[20, 20, 20],
        dataslices=None,
        usempi=False,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """"Convert a directory of tifs to an hdf5 stack."""

    # Check if any output paths already exist.
    outpaths = {'out': h5path_out}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # Get the list of input filepaths.
    files = sorted(glob.glob(os.path.join(inputdir, regex)))

    # (plane, row, column) indexing to outlayout (where prc -> zyx).
    in2out = ['zyx'.index(o) for o in outlayout]

    # Get the properties of the output dataset.
    slices = utils.get_slice_objects_prc(dataslices, files)  # prc-order
    files = files[slices[0]]

    datashape_out_prc = (len(files),
                         len(range(*slices[1].indices(slices[1].stop))),
                         len(range(*slices[2].indices(slices[2].stop))))
    datashape_out = [datashape_out_prc[i] for i in in2out]

    datatype = datatype or io.imread(files[0]).dtype

    # Reshape the file list into a list of blockwise file lists.
    scs = chunksize[outlayout.index('z')]  # chunksize slice dimension
    files_blocks = zip(* [iter(files)] * scs)
    files_blocks += [tuple(files[-(len(files) % scs):])]

    # Get slice objects for every output block.
    slices_out_prc = [[slice(bnr * scs, bnr * scs + scs),
                       slice(0, datashape_out_prc[1]),
                       slice(0, datashape_out_prc[2])]
                      for bnr in range(0, len(files_blocks))]
    slices_out = [[sliceset_prc[i] for i in in2out]
                  for sliceset_prc in slices_out_prc]

    # Prepare for processing with MPI.
    series = np.array(range(0, len(files_blocks)), dtype=int)
    if usempi:
        mpi_info = utils.get_mpi_info()
        series = utils.scatter_series(mpi_info, series)[0]
        comm = mpi_info['comm']
    else:
        comm = None

    # Open the outputfile for writing and create the dataset or output array.
    h5file_out, ds_out = utils.h5_write(None, datashape_out, datatype,
                                        h5path_out,
                                        element_size_um=element_size_um,
                                        axislabels=outlayout,
                                        chunks=tuple(chunksize),
                                        usempi=usempi, comm=comm)

    # Write blocks of 2D images to the outputfile.
    for blocknr in series:
        ds_out = process_block(files_blocks[blocknr], ds_out,
                               slices, slices_out[blocknr], in2out)

    # Close the h5 files or return the output array.
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


def process_block(files, ds_out, slcs_in, slcs_out, in2out):
    """Read the block from 2D images and write to stack.

    NOTE: slices is in prc-order, slices_out is in outlayout-order
    """

    datashape_block = (len(files),
                       len(range(*slcs_in[1].indices(slcs_in[1].stop))),
                       len(range(*slcs_in[2].indices(slcs_in[2].stop))))
    block = np.empty(datashape_block)
    for i, fpath in enumerate(files):
        block[i, :, :] = io.imread(fpath)[slcs_in[1], slcs_in[2]]

    ds_out[slcs_out[0], slcs_out[1], slcs_out[2]] = block.transpose(in2out)

    return ds_out


if __name__ == "__main__":
    main(sys.argv)
