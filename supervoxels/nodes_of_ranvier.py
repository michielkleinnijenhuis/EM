#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
import pickle

from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion, ball, watershed
from skimage.measure import regionprops
from scipy.ndimage.morphology import binary_dilation as scipy_binary_dilation

try:
    from mpi4py import MPI
except:
    print("mpi4py could not be loaded")


# TODO: write elsize and axislabels
def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', default=['_labelMA', '/stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('-s', '--min_labelsize', type=int, default=None,
                        help='...')
    parser.add_argument('--data', nargs=2, default=['', 'stack'],
                        help='...')
    parser.add_argument('--maskMM', nargs=2, default=['_maskMM', 'stack'],
                        help='...')
    parser.add_argument('-S', '--mask_sides', nargs=2,
                        default=['_maskDS_invdil', 'stack'],
                        help='...')
    parser.add_argument('-g', '--generate_mask', action='store_true',
                        help='...')
    parser.add_argument('-m', '--merge_method', default='neighbours',
                        help='...')
    parser.add_argument('-r', '--searchradius', nargs=3, type=int,
                        default=[100, 30, 30],
                        help='...')
    parser.add_argument('-o', '--outpf', nargs=2,
                        default=['_labelMA_2Dcore_fw_', 'stack'],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    outpf = args.outpf
    labelvolume = args.labelvolume
    min_labelsize = args.min_labelsize
    data = args.data
    maskMM = args.maskMM
    mask_sides = args.mask_sides
    generate_mask = args.generate_mask
    merge_method = args.merge_method
    searchradius = args.searchradius

    # open datasets
    lname = dset_name + labelvolume[0] + '.h5'
    lpath = os.path.join(datadir, lname)
    lfile = h5py.File(lpath, 'r')
    lstack = lfile[labelvolume[1]]
    labels = lstack[:,:,:]
    elsize, al = get_h5_attributes(lstack)
    lfile.close()

    if min_labelsize is not None:
        labels, _ = filter_on_size(datadir, dset_name, outpf,
                                   labels, min_labelsize)

    ulabels = np.unique(labels)
    maxlabel = np.amax(ulabels)
    print("number of labels in labelvolume: %d" % maxlabel)
    labelset = set(ulabels)

    # get a mask of the sides of the dataset
    sidesmask = get_mask(datadir, dset_name, mask_sides, elsize, al,
                         generate_mask)
    # get the labelsets that touch the borders
    ls_bot = set(np.unique(labels[0,:,:]))
    ls_top = set(np.unique(labels[-1,:,:]))
    ls_sides = set(np.unique(labels[sidesmask]))
    ls_border = ls_bot | ls_top | ls_sides
    ls_centre = labelset - ls_border
    # get the labels that do not touch the border twice
    ls_bts = (ls_bot ^ ls_top) ^ ls_sides
    ls_tbs = (ls_top ^ ls_bot) ^ ls_sides
    ls_sbt = (ls_sides ^ ls_bot) ^ ls_top
    # get the labels that do not pass through the volume
    ls_nt = ls_centre | ls_bts | ls_tbs | ls_sbt

    fw = np.zeros(maxlabel + 1, dtype='i')
    for l in ls_nt:
        fw[l] = l
    labels_nt = fw[labels]

    gname = dset_name + outpf[0] + '.h5'
    gpath = os.path.join(datadir, gname)
    gfile = h5py.File(gpath, 'w')
    outds = gfile.create_dataset(outpf[1], labels.shape,
                                 dtype='uint32', compression='gzip')
    write_h5_attributes(gfile[outpf[1]], elsize, al)
    outds[:,:,:] = labels_nt
    gfile.close()

    # find connection candidates
    if merge_method == 'neighbours':
        labelsets = merge_neighbours(labels, labels_nt, overlap_thr=20)

    elif merge_method == 'conncomp':
        labelsets = merge_conncomp(labels, labels_nt)

    elif merge_method == 'watershed':
        labelsets, filled = merge_watershed(labels, labels_nt,
                                            datadir, dset_name, data, maskMM,
                                            min_labelsize, searchradius)
        gname = dset_name + outpf[0] + '_filled.h5'
        gpath = os.path.join(datadir, gname)
        gfile = h5py.File(gpath, 'w')
        outds = gfile.create_dataset(outpf[1], labels.shape,
                                     dtype='uint32', compression='gzip')
        outds[:,:,:] = forward_map(np.array(fw), filled, labelsets)
        write_h5_attributes(gfile[outpf[1]], elsize, al)
        gfile.close()
    else:
        return

    filename = dset_name + outpf[0] + "_automerged"
    filestem = os.path.join(datadir, filename)
    write_labelsets(labelsets, filestem, filetypes=['txt', 'pickle'])

    gpath = filestem + '.h5'
    gfile = h5py.File(gpath, 'w')
    outds = gfile.create_dataset(outpf[1], labels.shape,
                                 dtype='uint32', compression='gzip')
    write_h5_attributes(gfile[outpf[1]], elsize, al)
    outds[:,:,:] = forward_map(np.array(fw), labels, labelsets)
    gfile.close()

# ========================================================================== #
# function defs
# ========================================================================== #


def get_mask(datadir, dset_name, maskfile, elsize, al,
             generate=False, masktype='invdil'):
    """Read or generate a mask."""

    mname = dset_name + maskfile[0] + '.h5'
    mpath = os.path.join(datadir, mname)
    m = h5py.File(mpath, 'r')
    mstack = m[maskfile[1]]
    mask = mstack[:,:,:].astype('bool')
    m.close()

    if generate:
        if masktype == 'ero':
            mask = binary_erosion(mask, ball(3))
        elif masktype == 'invdil':
#             invdilmask = binary_dilation(~maskDS, ball(3))
            mask = scipy_binary_dilation(~mask, iterations=1, border_value=0)
            mask[:4,:,:] = 0
            mask[-4:,:,:] = 0

        writeh5(mask, datadir, dset_name + maskfile[0] + '_' + masktype,
                dtype='uint8', element_size_um=elsize, axislabels=al)

    return mask


def filter_on_size(datadir, dset_name, outpf, labels, min_labelsize,
                   apply_to_labels=False, write_to_file=True):
    """Filter small labels from a volume; write the set to file."""

    areas = np.bincount(labels.ravel())
    fwmask = areas < min_labelsize

    ls_small = set([l for sl in np.argwhere(fwmask) for l in sl])

    if write_to_file:
        labelsets = {0: ls_small}
        filename = dset_name + outpf[0] + "_smalllabels"
        filestem = os.path.join(datadir, filename)
        write_labelsets(labelsets, filestem, filetypes=['txt', 'pickle'])

    if apply_to_labels:
        smalllabelmask = np.array(fwmask, dtype='bool')[labels]
        labels[smalllabelmask] = 0

    return labels, ls_small


def write_labelsets(labelsets, filestem, filetypes='txt'):
    """Write labelsets to file."""

    if 'txt' in filetypes:
        filepath = filestem + '.txt'
        write_labelsets_to_txt(labelsets, filepath)
    if 'pickle' in filetypes:
        filepath = filestem + '.pickle'
        with open(filepath, 'wb') as file:
            pickle.dump(labelsets, file)


def write_labelsets_to_txt(labelsets, filepath):
    """Write labelsets to a textfile."""

    with open(filepath, 'wb') as file:
        for lsk, lsv in labelsets.items():
            file.write("%8d: " % lsk)
            ls = sorted(list(lsv))
            for l in ls:
                file.write("%10d" % l)
            file.write('\n')


def find_region_coordinates(direction, labels, prop, searchradius):
    """Find coordinates of a box bordering a partial label."""

    # prop.bbox is in half-open interval
    if direction == 'around':
        z = max(0, int(prop.bbox[0]) - searchradius[0])
        Z = min(labels.shape[0], int(prop.bbox[3]) + searchradius[0])
        y = max(0, int(prop.bbox[1]) - searchradius[1])
        Y = min(labels.shape[1], int(prop.bbox[4]) + searchradius[1])
        x = max(0, int(prop.bbox[2]) - searchradius[2])
        X = min(labels.shape[2], int(prop.bbox[5]) + searchradius[2])
        return (x, X, y, Y, z, Z)
    elif direction == 'down':
        bs = int(prop.bbox[0])
        z = max(0, bs - searchradius[0])
        Z = bs
    elif direction == 'up':
        bs = int(prop.bbox[3]) - 1
        z = bs
        Z = min(labels.shape[0], bs + searchradius[0])

    labels_slc = np.copy(labels[bs,:,:])
    labels_slc[labels_slc != prop.label] = 0
    rp_bs = regionprops(labels_slc)
    ctrd = rp_bs[0].centroid

    y = max(0, int(ctrd[0]) - searchradius[1])
    Y = min(labels.shape[1], int(ctrd[0]) + searchradius[1])
    x = max(0, int(ctrd[1]) - searchradius[2])
    X = min(labels.shape[2], int(ctrd[1]) + searchradius[2])

    return (x, X, y, Y, z, Z)


def merge_neighbours(labels, labels_nt, overlap_thr=20):
    """Find candidates for label merge based on overlapping faces."""

    labelsets = {}
    rp_nt = regionprops(labels_nt)

    for prop in rp_nt:

        C = find_region_coordinates('around', labels, prop, [1,1,1])
        x, X, y, Y, z, Z = C

        imregion = labels_nt[z:Z,y:Y,x:X]
        labelmask = imregion == prop.label
        boundary = binary_dilation(labelmask) - labelmask

        counts = np.bincount(imregion[boundary])
        label_neighbours = np.argwhere(counts > overlap_thr)
        label_neighbours = [l for ln in label_neighbours for l in ln]
        if len(label_neighbours) > 1:
            labelset = set([prop.label] + label_neighbours[1:])
            labelsets = classify_label(labelsets, labelset, prop.label)

    return labelsets


def merge_conncomp(labels, labels_nt):
    """Find candidates for label merge based on connected components."""

    labelsets = {}
    labelmask = labels_nt != 0
    labels_connected = label(labelmask, connectivity=1)
    rp = regionprops(labels_connected, labels_nt)
    for prop in rp:
        counts = np.bincount(prop.intensity_image[prop.image])
        labelset = set(list(np.flatnonzero(counts)))
        if len(counts) > 1:
            labelsets = classify_label(labelsets, labelset, prop.label)

    return labelsets


def merge_watershed(labels, labels_nt,
                    datadir, dset_name, data, maskMM,
                    min_labelsize=0, searchradius=[100, 30, 30]):
    """Find candidates for label merge based on watershed."""

    labelsets = {}

    rp_nt = regionprops(labels_nt)
    labels_filled = np.copy(labels_nt)

    dname = dset_name + data[0] + '.h5'
    dpath = os.path.join(datadir, dname)
    d = h5py.File(dpath, 'r')
    dstack = d[data[1]]
    ws_data = dstack[:,:,:]

    mmname = dset_name + maskMM[0] + '.h5'
    mmpath = os.path.join(datadir, mmname)
    mm = h5py.File(mmpath, 'r')
    mmstack = mm[maskMM[1]]
    ws_mask = mmstack[:,:,:].astype('bool')

    for prop in rp_nt:
        # investigate image region above and below bbox:
        for direction in ['down', 'up']:

            C = find_region_coordinates(direction, labels,
                                        prop, searchradius)
            x, X, y, Y, z, Z = C
            if ((z == 0) or (z == labels.shape[0] - 1)):
                continue

            imregion = labels_nt[z:Z,y:Y,x:X]
            labels_in_region = np.unique(imregion)
            if len(labels_in_region) < 2:  # always label 0 and prop.label
                continue

            labelsets, wsout = find_candidate_ws(direction, labelsets, prop,
                                                 imregion,
                                                 ws_data[z:Z, y:Y, x:X],
                                                 ws_mask[z:Z, y:Y, x:X],
                                                 min_labelsize)

            if wsout is not None:
                labels_filled[z:Z,y:Y,x:X] = np.copy(wsout)

    return labelsets, labels_filled


def find_candidate_ws(direction, labelsets, prop, imregion,
                      data, maskMM, min_labelsize):
    """Find a merge candidate by watershed overlap."""

    wsout = None

    # seeds are in the borderslice, 
    # with the current label as prop.label (watershedded to fill the full axon),
    # the maskMM as background, and the surround as negative label
    if direction == 'down':
        idx = -1
    elif direction == 'up':
        idx = 0
    seeds = np.zeros_like(imregion)
    seeds[idx,:,:] = watershed(-data[idx,:,:], imregion[idx,:,:],
                               mask=~maskMM[idx,:,:])
    seeds[idx,:,:][seeds[idx,:,:] != prop.label] = -1
    seeds[idx,:,:][maskMM[idx,:,:]] = 0

    # do the watershed
    ws = watershed(-data, seeds, mask=~maskMM)

    rp_ws = regionprops(ws, imregion) # no 0 in rp
    labels_ws = [prop_ws.label for prop_ws in rp_ws]
    try:
        idx = labels_ws.index(prop.label)
    except ValueError:
        pass
    else:
        counts = np.bincount(imregion[rp_ws[idx].image])
        if len(counts) > 1:
            # select the largest candidate overlapping the watershed
            candidate = np.argmax(counts[1:]) + 1
            # only select it if it a proper region
            if counts[candidate] > min_labelsize:
                labelset = set([prop.label, candidate])
                labelsets = classify_label(labelsets, labelset, prop.label)
                wsout = ws
                mask = ws != prop.label
                wsout[mask] = imregion[mask]

    return labelsets, wsout


def classify_label(labelsets, labelset, lskey=None):
    """Add labels to a labelset or create a new set."""

    found = False
    for lsk, lsv in labelsets.items():
        for l in labelset:
            if l in lsv:
                labelsets[lsk] = lsv | labelset
                found = True
                return labelsets
    if not found:
        if lskey is None:
            lskey = min(labelset)
        labelsets[lskey] = labelset

    return labelsets


def forward_map(fw, labels, labelsets):
    """Map all labelset in value to key."""

    for lsk, lsv in labelsets.items():
        lsv = sorted(list(lsv))
        for l in lsv:
            fw[l] = lsk

    fwmapped = fw[labels]

    return fwmapped


def scatter_series(n, comm, size, rank, SLL):
    """Scatter a series of jobnrs over processes."""

    nrs = np.array(range(0, n), dtype=int)
    local_n = np.ones(size, dtype=int) * n / size
    local_n[0:n % size] += 1
    local_nrs = np.zeros(local_n[rank], dtype=int)
    displacements = tuple(sum(local_n[0:r]) for r in range(0, size))
    comm.Scatterv([nrs, tuple(local_n), displacements,
                   SLL], local_nrs, root=0)

    return local_nrs, tuple(local_n), displacements


def loadh5(datadir, dname, fieldname='stack', dtype=None):
    """Load a h5 stack."""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]

    element_size_um, axislabels = get_h5_attributes(f[fieldname])

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """Write a h5 stack."""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    write_h5_attributes(g[fieldname], element_size_um, axislabels)

    g.close()


def get_h5_attributes(stack):
    """Get attributes from a stack."""

    element_size_um = axislabels = None

    if 'element_size_um' in stack.attrs.keys():
        element_size_um = stack.attrs['element_size_um']

    if 'DIMENSION_LABELS' in stack.attrs.keys():
        axislabels = stack.attrs['DIMENSION_LABELS']

    return element_size_um, axislabels


def write_h5_attributes(stack, element_size_um=None, axislabels=None):
    """Write attributes to a stack."""

    if element_size_um is not None:
        stack.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, l in enumerate(axislabels):
            stack.dims[i].label = l


if __name__ == "__main__":
    main(sys.argv[1:])
