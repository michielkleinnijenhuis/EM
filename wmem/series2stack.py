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
import os
import glob
from math import ceil

import numpy as np
from skimage import io

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py could not be loaded")

try:
    import DM3lib as dm3
except ImportError:
    print("dm3lib could not be loaded")

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
        args.outputformats,
        args.outputpath,
        args.save_steps,
        args.protective,
        )


def series2stack(
        inputdir,
        regex='*.tif',
        element_size_um=[None, None, None],
        outlayout='zyx',
        datatype='',
        chunksize=[20, 20, 20],
        dataslices=None,
        usempi=False,
        outputformats=['.h5'],
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """"Convert a directory of tifs to an hdf5 stack."""

    # Check if any output paths already exist.
    if '.h5' in outputformats:
        outpaths = {'out': outputpath}
        status = utils.output_check(outpaths, save_steps, protective)
        if status == "CANCELLED":
            return

    # Get the list of input filepaths.
    files = sorted(glob.glob(os.path.join(inputdir, regex)))

    # Get some metadata from the inputfiles
    zyxdims, datatype, element_size_um = get_metadata(files,
                                                      datatype,
                                                      outlayout,
                                                      element_size_um)

    # (plane, row, column) indexing to outlayout (where prc -> zyx).
    in2out = ['zyx'.index(o) for o in outlayout]

    # Get the properties of the output dataset.
    slices = utils.get_slice_objects_prc(dataslices, zyxdims)  # prc-order
    files = files[slices[0]]
    datashape_out_prc = (len(files),
                         len(range(*slices[1].indices(slices[1].stop))),
                         len(range(*slices[2].indices(slices[2].stop))))
    datashape_out = [datashape_out_prc[i] for i in in2out]

    # Reshape the file list into a list of blockwise file lists.
    scs = chunksize[outlayout.index('z')]  # chunksize slice dimension
    files_blocks = zip(* [iter(files)] * scs)
    rem = len(files) % scs
    if rem:
        files_blocks += [tuple(files[-rem:])]

    # Get slice objects for every output block.
    slices_out_prc = [[slice(bnr * scs, bnr * scs + scs),
                       slice(0, datashape_out_prc[1]),
                       slice(0, datashape_out_prc[2])]
                      for bnr in range(0, len(files_blocks))]
    slices_out = [[sliceset_prc[i] for i in in2out]
                  for sliceset_prc in slices_out_prc]

    # Prepare for processing with MPI.
    mpi_info = utils.get_mpi_info(usempi)
    series = np.array(range(0, len(files_blocks)), dtype=int)
    if mpi_info['enabled']:
        series = utils.scatter_series(mpi_info, series)[0]

    # Open the outputfile for writing and create the dataset or output array.
    if '.h5' in outputformats:
        h5file_out, ds_out = utils.h5_write(None, datashape_out, datatype,
                                            outputpath,
                                            element_size_um=element_size_um,
                                            axislabels=outlayout,
                                            chunks=tuple(chunksize),
                                            comm=mpi_info['comm'])
        outdir = os.path.dirname(outputpath.split('.h5')[0])
    else:
        ds_out = None
        outdir = outputpath

    # Write blocks of 2D images to the outputfile(s).
    for blocknr in series:
        if '.h5' in outputformats:
            ds_out = process_block(files_blocks[blocknr], ds_out,
                                   slices, slices_out[blocknr], in2out,
                                   outputformats, outdir)
        else:
            process_slices(files_blocks[blocknr],
                           slices, slices_out[blocknr],
                           outputformats, outdir, datatype)

    # Close the h5 files or return the output array.
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out
    except UnboundLocalError:
        pass


def process_block(files, ds_out, slcs_in, slcs_out, in2out,
                  outputformats=['.h5'], outdir=''):
    """Read the block from 2D images and write to stack.

    NOTE: slices is in prc-order, slices_out is in outlayout-order
    """

    datashape_block = (
        len(files),
        len(range(*slcs_in[1].indices(slcs_in[1].stop))),
        len(range(*slcs_in[2].indices(slcs_in[2].stop))),
        )
    block = np.empty(datashape_block)
    for i, fpath in enumerate(files):
        im = get_image(fpath, ds_out.dtype)
        block[i, :, :] = im[slcs_in[1], slcs_in[2]]

    for ext in outputformats:
        if ext == '.h5':
            ds_out[slcs_out[0],
                   slcs_out[1],
                   slcs_out[2]] = block.transpose(in2out)
        elif ext in ('.tif', '.png', '.jpg'):
            utils.write_to_img(outdir, block,
                               'zyx', nzfills=5, ext=ext,
                               slcoffset=slcs_out[0].start)

    return ds_out


def process_slices(files, slcs_in, slcs_out,
                   outputformats=['.tif'], outdir='',
                   datatype='uint16'):
    """Read the block from 2D images and write to stack.

    NOTE: slices is in prc-order, slices_out is in outlayout-order
    """

    for i, fpath in enumerate(files):
        slcno = slcs_out[0].start + i
        im = get_image(fpath, datatype)

        for ext in outputformats:
            savedir = os.path.join(outdir, ext[1:])
            utils.mkdir_p(savedir)
            data = im[slcs_in[1], slcs_in[2]]
            if ext != '.tif':
                data = utils.normalize_data(data)[0]
                # FIXME: normalize over full dataset
            fname = '{:05d}{}'.format(slcno, ext)
            filepath = os.path.join(savedir, fname)
            io.imsave(filepath, data)


def get_image(fpath, datatype):

    if fpath.endswith('.dm3'):
        dm3f = dm3.DM3(fpath, debug=0)
        im = dm3f.imagedata
    else:
        im = io.imread(fpath)

    return im.astype(datatype)


def get_metadata(files, datatype, outlayout, elsize):

    # derive the stop-values from the image data if not specified
    if files[0].endswith('.dm3'):

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

    else:
        yxdims = io.imread(files[0]).shape

        alt_dtype = io.imread(files[0]).dtype

    zyxdims = [len(files)] + list(yxdims)

    datatype = datatype or alt_dtype

    elsize = [0 if el is None else el for el in elsize]

    return zyxdims, datatype, elsize


if __name__ == "__main__":
    main(sys.argv)
