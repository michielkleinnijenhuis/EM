#!/usr/bin/env python

"""Label connected components.

"""

import sys
import argparse
import os
import pickle

import numpy as np

from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing
from skimage.morphology import binary_dilation

from wmem import parse, utils


def main(argv):
    """Label connected components."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_connected_components_clf(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    if args.mode == 'train':

        CC_clf_train(
            args.inputfile,
            args.outputfile,
            args.save_steps,
            args.protective,
            )

    elif args.mode == 'test':

        CC_clf_test(
            args.classifierpath,
            args.scalerpath,
            args.basename,
            args.map_propnames,
            args.max_intensity_mb,
            args.max_area,
            args.inputfile,
            args.outputfile,
            args.maskfile,
            args.save_steps,
            args.protective,
            )

    if args.mode == 'map':

        CC_clf_map(
            args.inputfile,
            args.outputfile,
            args.save_steps,
            args.protective,
            )


def CC_clf_gtgen(
        h5path_in,
        h5path_mask,
        outputfile='',
        ):
    """Label connected components in a 3D stack."""

    # get 2D labels and mask of identified 3D MA compartment
    labels = utils.h5_load(h5path_in, load_data=True)[0]
    maskMA = utils.h5_load(h5path_mask, load_data=True)[0]
#     mask = np.zeros_like(labels, dtype='bool')
#     m = h5py.File(os.path.join(datadir, maskfile), 'r')
#     mask[:, :-1, :-1] = m[maskdset][:].astype('bool')
#     m.close()

    # split the labels in MA and notMA
    labelsALL = np.unique(labels)
    maskdil = binary_dilation(maskMA)
    labelsMA = np.unique(labels[maskdil])
    labelsNOTMA = np.unique(labels[~maskdil])

    # filter labels that are split between compartments
    labelsTRUE = set(labelsMA) - set(labelsNOTMA)
    labelsFALSE = set(labelsALL) - set(labelsMA)
    print(len(labelsTRUE), len(labelsFALSE))

    # generate final ground truth forward map
    y = np.zeros_like(labelsALL, dtype='bool')
    for l in labelsTRUE:
        y[l] = True
    y[0] = False
    np.save(outputfile, y)


def CC_clf_train(
        clfpath,
        scalerpath,
        groundtruthpath,
        basename,
        map_propnames,
        save_steps=False,
        protective=False,
        ):
    """Label connected components in a 3D stack."""

    # check output paths
    outpaths = {'out': clfpath, 'scaler': scalerpath}
    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # Load the ground truth labels.
    y_train = np.load(groundtruthpath)
    y_train = y_train[1:]  # not the background label

    # Load the training data.
    X_train = np.zeros([y_train.shape[0], len(map_propnames) - 1])
    for i, propname in enumerate(map_propnames[1:]):
        nppath = '{}_{}.npy'.format(basename, propname)
        X_train[:, i] = np.load(nppath)[1:]  # not the background label

    # Scale the training data.
    scaler = preprocessing.MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit the support vector classifier.
    clf = svm.SVC()
    clf.fit(X_train_scaled, y_train)

    # Write the classifier and scaler.
    joblib.dump(clf, clfpath)
    joblib.dump(scaler, scalerpath)


def CC_clf_test(
        clfpath,
        scalerpath,
        basename,
        map_propnames,
        thr_mi=0,
        thr_area=0,
        h5path_in='',
        h5path_out='',
        h5path_mask='',
        save_steps=False,
        protective=False,
        ):
    """Label connected components in a 3D stack."""

    # check output paths
    predpath = '{}_{}.npy'.format(basename, 'pred')
    predpath_thr = '{}_{}.npy'.format(basename, 'pred_thr')

    outpaths = {'pred': predpath}
    if h5path_out:
        outpaths['out'] = h5path_out
        if h5path_mask:
            outpaths['maskMA'] = h5path_mask
        elif save_steps:
            root, ds_main = outpaths['out'].split('.h5')
            grpname = ds_main + "_steps"
            outpaths['maskMA'] = os.path.join(root + '.h5' + grpname, 'maskMA')
    if thr_mi or thr_area:
        outpaths['pred_thr'] = predpath_thr

    status = utils.output_check(outpaths, save_steps, protective)
    if status == "CANCELLED":
        return

    # Load the scaler and classifier.
    scaler = joblib.load(scalerpath)
    clf = joblib.load(clfpath)

    # Scale the test data.
    tmppath = '{}_{}.npy'.format(basename, map_propnames[0])
    tmp = np.load(tmppath)[1:]
    X_test = np.zeros([tmp.shape[0], len(map_propnames) - 1])
    for i, propname in enumerate(map_propnames[1:]):
        nppath = '{}_{}.npy'.format(basename, propname)
        X_test[:, i] = np.load(nppath)[1:]
    X_test_scaled = scaler.transform(X_test)

    # Predict the test data.
    pred = clf.predict(X_test_scaled)
    fw = np.insert(pred, 0, [False])  # reinsert the background label

    # Save the results.
    np.save(predpath, fw)

    # Apply additional criteria.
    fw = apply_additional_criteria(basename, fw, thr_mi, thr_area)

    # Map the predicted labels to a volume.
    if (h5path_in and h5path_out):
        map_to_volume(fw, h5path_in, h5path_out, h5path_mask)


def CC_clf_map(
        h5path_in,
        h5path_out,
        predpath,
        h5path_mask='',
        ):
    """Map the prediction to a labelvolume."""

    fw = np.load(predpath)
    map_to_volume(fw, h5path_in, h5path_out, h5path_mask)


def apply_additional_criteria(basename, fw, thr_mi=0.8, thr_area=3000):
    """Apply additional criteria to forward map."""

    if thr_mi:
        nppath = '{}_{}.npy'.format(basename, 'label')
        fw_all = np.load(nppath)

        nppath = '{}_{}.npy'.format(basename, 'mean_intensity')
        fwf = np.load(nppath)
        m = fwf > thr_mi
        fw[m] = fw_all[m]  # always include

    if thr_area:
        nppath = '{}_{}.npy'.format(basename, 'area')
        fwf = np.load(nppath)
        m = fwf > thr_area
        fw[m] = 0  # always exclude

    predpath_thr = '{}_{}.npy'.format(basename, 'pred_thr')
    np.save(predpath_thr, fw)

    return fw


def map_to_volume(fw, h5path_in, h5path_out, h5path_mask=''):
    """Map the prediction to a labelvolume."""

    h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
    a = ds_in[:]
    h5file_in.close()

    # myelinated axon labels
    h5file_out, ds_out = utils.h5_write(None, a.shape, a.dtype,
                                        h5path_out,
                                        element_size_um=elsize,
                                        axislabels=axlab)

    mask = fw[a]
    a[~mask] = 0
    ds_out[:] = a
    h5file_out.close()

    # myelinated axon mask
    if not h5path_mask:
        return

    h5file_out, ds_out = utils.h5_write(None, a.shape, 'bool',
                                        h5path_mask,
                                        element_size_um=elsize,
                                        axislabels=axlab)
    ds_out[:] = a.astype('bool')
    h5file_out.close()


if __name__ == "__main__":
    main(sys.argv[1:])
