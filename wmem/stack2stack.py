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
    im = utils.get_image(image_in, comm=mpi.comm, dataslices=dataslices, load_data=False)

    if dset_name:
        im.slices = utils.dset_name2slices(dset_name, blockoffset,
                                           axlab=inlayout,
                                           shape=im.dims)
    props = im.get_props(elsize=elsize, chunks=chunksize, dtype=datatype)

    # Load the data
    data = im.slice_dataset(squeeze=False)

    # Convert datatype  # TODO: proper general writing astype
    if uint8conv:
        from skimage.util.dtype import convert
        #from skimage.util.dtype import convert, img_as_float, img_as_ubyte
        # if data.dtype == np.float:
        #     print('got floats')
        #     data = img_as_float(data)
        #data = data.astype('uint32')
        data = convert(data, np.dtype(datatype), force_copy=False)
        props['dtype'] = datatype

    outlayout = outlayout or props['axlab']
    in2out = [props['axlab'].index(l) for l in outlayout]
#     props['chunks'] = list(props['chunks'])
#     print(props)
    props, data = remove_singleton(im, props, data, outlayout)
    props, data = permute_axes(im, props, data, in2out)

    trans = [es * slc.start for es, slc in zip(im.elsize, im.slices)]
    im.close()

    # Open the outputfile for writing and create the dataset or output array.
#     props['dtype'] = datatype or props['dtype']
#     props['chunks'] = chunksize or props['chunks']
#     props['elsize'] = elsize or props['elsize']

    mo = Image(outputpath, **props)
    mo.create(comm=mpi.comm)
    mo.slices = None
    mo.set_slices()

    if mo.format == '.nii':
        # TODO: properly implement translation of cutouts
        mat = mo.get_transmat()
        if im.slices is not None:
            trans = np.array(trans)[in2out]
            mat[0][3] = trans[0]
            mat[1][3] = trans[1]
            mat[2][3] = trans[2]
        mo.nii_write_mat(data, mo.slices, mat)
    else:
        mo.write(data=data)

    mo.close()

    return mo


def remove_singleton(im, props, data, outlayout):

    # TODO: singleton-check
    for al in props['axlab']:
        if al not in outlayout:
            dim = props['axlab'].index(al)
            props = im.remove_singleton_props(props, dim)
            data = np.squeeze(data, dim)

    return props, data


def permute_axes(im, props, data, in2out):

    for prop in ['elsize', 'shape', 'chunks', 'axlab', 'slices']:
        props[prop] = utils.transpose(props[prop], in2out)
    data = np.transpose(data, in2out)

    return props, data


if __name__ == "__main__":
    main(sys.argv)
