#!/usr/bin/env python

"""Convert dm3 files to other image formats.

"""

import os
import sys
import argparse

import numpy as np
# from PIL import Image

import DM3lib as dm3
from glob import glob

from wmem import parse, utils


def main(argv):
    """Convert dm3 files to other image formats."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_convert_dm3(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    convert_dm3(
        args.inputfile,
        args.masks,
        args.seedimage,
        args.seed_size,
        args.lower_threshold,
        args.upper_threshold,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def convert_dm3(
        filepaths,
        outdir='',
        save_steps=False,
        protective=False,
        ):
    """Convert dm3 files to other image formats."""

    # check output paths
    outpaths = {'out': outdir}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    if not os.path.exists(outdir):
        os.makedirs(savedir)

    for filepath in filepaths:

        filename = os.path.split(filepath)[1]
        fileref = os.path.splitext(filename)[0]

        im, dm3f = read_dm3_as_im(filepath)

        if dumptags:
            dm3f.dumpTags(outdir)

        if '.tif' in outexts:
            outfilepath = os.path.join(outdir, fileref + '.tif')
            im.save(outfilepath)



def read_dm3_as_ndimage(filepath):
    """Read a .dm3 file."""

    filename = os.path.split(filepath)[1]
    fileref = os.path.splitext(filename)[0]
    dm3f = dm3.DM3(filepath, debug=0)
    im = dm3f.imagedata

    return im, dm3f


def convert_to_8bit(im):
    """"""

    # - normalize image for conversion to 8-bit
    aa_norm = aa.copy()
    # -- apply cuts (optional)
    if cuts[0] != cuts[1]:
        aa_norm[ (aa <= min(cuts)) ] = float(min(cuts))
        aa_norm[ (aa >= max(cuts)) ] = float(max(cuts))
    # -- normalize
    aa_norm = (aa_norm - np.min(aa_norm)) / (np.max(aa_norm) - np.min(aa_norm))
    # -- scale to 0--255, convert to (8-bit) integer
    aa_norm = np.uint8(np.round( aa_norm * 255 ))
    # - save as <imformat>
    im_dsp = Image.fromarray(aa_norm)
    im_dsp.save(os.path.join(savedir, fileref + imformat))


if __name__ == "__main__":
    main(sys.argv[1:])
