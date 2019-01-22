#!/usr/bin/env python

"""Apply mapping of labelsets to a labelvolume.

"""

import sys
import argparse

import numpy as np


from wmem import parse, utils, LabelImage


def main(argv):
    """Apply mapping of labelsets to a labelvolume."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_agglo_from_labelsets(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    agglo_from_labelsets(
        args.inpufile,
        args.labelset_files,
        args.fwmap,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def agglo_from_labelsets(
        image_in,
        labelset_files='',
        fwmap='',
        outputpath='',
        save_steps=False,
        protective=False,
        ):
    """Apply mapping of labelsets to a labelvolume."""

    im = utils.get_image(image_in, imtype='Label')

    outpaths = get_outpaths(outputpath, save_steps)
    props = im.get_props(protective=protective)
    mos = {}
    for stepname, outpath in outpaths.items():
        mos[stepname] = LabelImage(outpath, **props)
        mos[stepname].create()

    # read labelsets from file
    for lsfile in labelset_files:
        labelsets = utils.read_labelsets(lsfile)

    # get labelsets from forward map
    if fwmap:
        fw = np.load(fwmap)
        for lsk, lsv in labelsets.items():
            remapped = []
            for l in lsv:
                idx = np.argwhere(fw == l)
                if idx.size:
                    remapped.append(idx[0][0])
            labelsets[lsk] = set(remapped)

    mos['out'].ds[:] = im.forward_map(labelsets=labelsets,
                                      from_empty=True)
    if save_steps:
        mos['del'].ds[:] = im.forward_map(labelsets=labelsets,
                                          delete_labelsets=True)
        mos['all'].ds[:] = im.forward_map(labelsets=labelsets)

    im.close()
    for _, mo in mos.items():
        mo.close()

    return mos['out']


def get_outpaths(h5path_out, save_steps):

    outpaths = {'out': h5path_out}
    if save_steps:
        outpaths['del'] = ''
        outpaths['all'] = ''
    outpaths = utils.gen_steps(outpaths, save_steps)

    return outpaths


if __name__ == "__main__":
    main(sys.argv[1:])
