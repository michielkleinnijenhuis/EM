#!/usr/bin/env python

"""Calculate SLIC supervoxels.

"""

import os
import sys
import argparse

import numpy as np
from skimage import segmentation
import maskslic as seg

from wmem import parse, utils, LabelImage


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

    mpi_info = utils.get_mpi_info(usempi)

    # Open the inputfile for reading.
    im = utils.get_image(image_in, comm=mpi_info['comm'],
                         dataslices=dataslices)
    mask = utils.get_image(masks[0], comm=mpi_info['comm'],
                           dataslices=dataslices)
    in2out_offset = -np.array([slc.start for slc in im.slices])

    # Determine the properties of the output dataset.
    mo = LabelImage(outputpath, protective=protective,
                    **im.squeeze_channel())
    mo.create(comm=mpi_info['comm'])

    blocks = utils.get_blocks(im, blocksize, blockmargin, blockrange)
    series = utils.scatter_series(mpi_info, len(blocks))[0]

    spac = [es for es in np.absolute(mo.elsize)]

    for blocknr in series:
        im.slices = blocks[blocknr]['slices']
        mask.slices = blocks[blocknr]['slices']

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
        offset = 2 * n_segments * blocknr + 1
        segments[~maskdata] = segments[~maskdata] + offset
        print("Offsett blocknr {} by: {}".format(blocknr, offset))
        slices_out = im.get_offset_slices(in2out_offset)
        mo.write(data=segments, slices=slices_out)

    if mpi_info['enabled']:
        mpi_info['comm'].Barrier()
        if mpi_info['rank'] == 0 and mpi_info['size'] > 1:
            mo.ds[:] = segmentation.relabel_sequential(mo.ds[:].astype('i'))[0]
            print("Output was relabeled sequentially.")

    im.close()
    mo.close()

    return mo


if __name__ == "__main__":
    main(sys.argv[1:])

