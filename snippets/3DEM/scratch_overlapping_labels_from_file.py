def get_overlap(dset_info, side, fstack, gstack, granges,
                margin=[0, 0, 0], blocksize=[0, 0, 0], fullsize=[0, 0, 0]):
    """Return boundary slice of block and its neighbour."""

    x, X, y, Y, z, Z = granges
    nb_section = None

    if side == 'xmin':
        data_section = fstack[:, :, 0:margin[0]]

        if x > 0:
            dset_info['x'] = max(0, dset_info['x'] - blocksize[0])
            dset_info['X'] = dset_info['X'] - blocksize[0]
            gname = dataset_name(dset_info)
            g = h5py.File(os.path.join(dset_info['datadir'], gname), 'r')
            nb_section = g[z:Z, y:Y, :-margin[0]]

    elif side == 'xmax':
        data_section = fstack[:, :, :-margin[0]]

        if X < fullsize[0]:
            dset_info['x'] = dset_info['x'] + blocksize[0]
            dset_info['X'] = min(fullsize[0], dset_info['X'] + blocksize[0])
            gname = dataset_name(dset_info)
            g = h5py.File(os.path.join(dset_info['datadir'], gname), 'r')
            nb_section = g[z:Z, y:Y, 0:margin[0]]

    elif side == 'ymin':
        data_section = fstack[:, 0:margin[1], :]

        if y > 0:
            dset_info['y'] = max(0, dset_info['y'] - blocksize[1])
            dset_info['Y'] = dset_info['Y'] - blocksize[1]
            gname = dataset_name(dset_info)
            g = h5py.File(os.path.join(dset_info['datadir'], gname), 'r')
            nb_section = g[z:Z, :-margin[1], x:X]

    elif side == 'ymax':
        data_section = fstack[:, -1, :]

        if Y < fullsize[1]:
            dset_info['y'] = dset_info['y'] + blocksize[1]
            dset_info['Y'] = min(fullsize[1], dset_info['Y'] + blocksize[1])
            gname = dataset_name(dset_info)
            g = h5py.File(os.path.join(dset_info['datadir'], gname), 'r')
            nb_section = g[z:Z, 0:margin[1], x:X]

    elif side == 'zmin':
        data_section = fstack[0:margin[2], :, :]

        if z > 0:
            dset_info['z'] = max(0, dset_info['z'] - blocksize[2])
            dset_info['Z'] = dset_info['Z'] - blocksize[2]
            gname = dataset_name(dset_info)
            g = h5py.File(os.path.join(dset_info['datadir'], gname), 'r')
            nb_section = g[:-margin[2], y:Y, x:X]

    elif side == 'zmax':
        data_section = fstack[-1, :, :]

        if Z < fullsize[2]:
            dset_info['z'] = dset_info['z'] + blocksize[2]
            dset_info['Z'] = min(fullsize[2], dset_info['Z'] + blocksize[2])
            gname = dataset_name(dset_info)
            g = h5py.File(os.path.join(dset_info['datadir'], gname), 'r')
            nb_section = g[0:margin[2], y:Y, x:X]

    return data_section, nb_section
