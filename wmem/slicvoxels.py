#!/usr/bin/env python

"""Calculate SLIC supervoxels.

"""

import os
import sys
import argparse

import numpy as np
from skimage import segmentation
import maskslic as seg

from wmem import parse, utils, wmeMPI, LabelImage


def main(argv):
    """Calculate SLIC supervoxels."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_slicvoxels(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    slicvoxels(
        args.inputfile,
        args.dataslices,
        args.blocksize,
        args.blockmargin,
        args.blockrange,
        args.masks,
        args.slicvoxelsize,
        args.compactness,
        args.sigma,
        args.enforce_connectivity,
        args.outputfile,
        args.save_steps,
        args.protective,
        args.usempi & ('mpi4py' in sys.modules),
        )


def slicvoxels(
        image_in,
        dataslices=None,
        blocksize=[],
        blockmargin=[],
        blockrange=[],
        masks=[],
        slicvoxelsize=500,
        compactness=0.2,
        sigma=1,
        enforce_connectivity=False,
        outputpath='',
        save_steps=False,
        protective=False,
        usempi=False,
        ):
    """Calculate SLIC supervoxels."""

    mpi = wmeMPI(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices)
    mask = utils.get_image(masks[0], comm=mpi.comm, dataslices=dataslices)
    # TODO: string masks

    # Determine the properties of the output dataset.
    props = im.get_props(protective=protective, dtype='uint64', squeeze=True)
    mo = LabelImage(outputpath, **props)
    mo.create(comm=mpi.comm)
    in2out_offset = -np.array([slc.start for slc in mo.slices])

    # Prepare for processing with MPI.
    mpi.set_blocks(im, blocksize, blockmargin, blockrange)
    mpi.scatter_series()

    spac = [es for es in np.absolute(mo.elsize)]

    for i in mpi.series:
        block = mpi.blocks[i]

        im.slices = mask.slices = block['slices']

        blocksize = np.prod(np.array(list(im.slices2shape())))
        n_segments = int(blocksize / slicvoxelsize)

        data = im.slice_dataset()
        data = im.normalize_data(data)[0]
        maskdata = mask.slice_dataset().astype('bool')
        # NOTE: mask should still be 3D with 4D data
#         if im.get_ndim() == 4:
#             maskdata = np.repeat(maskdata[:, :, :, np.newaxis],
#                                  im.dims[3], axis=3)

        segments = seg.slic(data,
                            mask=~maskdata,
                            recompute_seeds=True,
                            n_segments=n_segments,
                            compactness=compactness,
                            spacing=spac,
                            sigma=sigma,
                            multichannel=len(data.shape) == 4,
                            convert2lab=False,
                            enforce_connectivity=enforce_connectivity)

        print("Number of supervoxels: {}".format(np.amax(segments) + 1))
        offset = 2 * n_segments * i + 1
        segments[~maskdata] = segments[~maskdata] + offset
        print("Offsett blocknr {} by: {}".format(i, offset))
        slices_out = im.get_offset_slices(in2out_offset)
        mo.write(data=segments, slices=slices_out)

    if mpi.enabled:
        mpi.comm.Barrier()
#         mo.set_maxlabel()
#         print(mpi_info['rank'], mo.maxlabel, mo.ulabels)
        if mpi.rank == 0 and mpi.size > 1:
            mo.ds[:] = segmentation.relabel_sequential(mo.ds[:])[0]
#             mo.set_maxlabel()
#             print('final', mo.maxlabel, mo.ulabels)
            print("Output was relabeled sequentially.")

    im.close()
    mo.close()

    return mo


if __name__ == "__main__":
    main(sys.argv[1:])

