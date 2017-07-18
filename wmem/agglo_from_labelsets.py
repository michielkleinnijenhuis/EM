#!/usr/bin/env python

"""Apply mapping of labelsets to a labelvolume.

"""

import sys
import argparse

import numpy as np

from wmem import parse, utils


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
        h5path_in,
        labelset_files='',
        fwmap='',
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Apply mapping of labelsets to a labelvolume."""

    # check output paths
    outpaths = {'out': h5path_out, 'deleted': '', 'added': ''}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # open data for reading
    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

    # open data for writing
    h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    ulabels = np.unique(ds_in)
    maxlabel = np.amax(ulabels)
    print("number of labels in watershed: %d" % maxlabel)

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

    # merge labelsets and return labelvolume with only the merged labels
    fw = np.zeros(maxlabel + 1, dtype='i')
    ds_out[:] = utils.forward_map(np.array(fw), ds_in, labelsets)

    # delete labelsets from the labelvolume
    fw = np.array([l if l in ulabels else 0
                   for l in range(0, maxlabel + 1)])
    fwmapped = utils.forward_map(np.array(fw), ds_in, labelsets,
                                 delete_labelsets=True)
    utils.save_step(outpaths, 'deleted', fwmapped, elsize, axlab)

    # merge labelsets and return labelvolume with all labels
    # TODO: could simply add the previous outputs?
    fw = np.array([l if l in ulabels else 0
                   for l in range(0, maxlabel + 1)])
    fwmapped = utils.forward_map(np.array(fw), ds_in, labelsets)
    utils.save_step(outpaths, 'added', fwmapped, elsize, axlab)

    # close and return
    h5file_in.close()
    try:
        h5file_out.close()
    except (ValueError, AttributeError):
        return ds_out


if __name__ == "__main__":
    main(sys.argv[1:])
