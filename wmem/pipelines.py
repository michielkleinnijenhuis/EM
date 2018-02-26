#!/usr/bin/env python

"""Standard pipelines for white matter electron microscopy segmentation.

"""

import os
from wmem import utils
from wmem.series2stack import series2stack
from wmem.downsample_slices import downsample_slices
from wmem.stack2stack import stack2stack
from wmem.prob2mask import prob2mask
from wmem.connected_components import CC_2D, CC_2Dfilter, CC_2Dprops
from wmem.separate_sheaths import separate_sheaths
from wmem.seg_stats import seg_stats


def pipeline_myelin2D(configfile='',
                      datadir='', datasets='',
                      options='', parameters='',
                      run_from='', run_upto='', run_only=''):
    """Run pipeline to segment myelinated axons in 2D.

    This function expects ... [TODO]
    """

    # retrieve requested configuration
    if configfile:
        ddir, dsets, opts, pars = utils.load_config(
            configfile,
            run_from=run_from, run_upto=run_upto, run_only=run_only,
            )
    datadir = datadir or ddir
    datasets = datasets or dsets
    options = options or opts
    parameters = parameters or pars

    if ((not datadir) or (not datasets) or
            (not options) or (not parameters)):
        print("missing info")
        return

    # simple mean intensity normalization over datasets
    if options['normalize_datasets']:
        utils.normalize_datasets(
            datadir, datasets,
            postfix=parameters['ds']['datapostfix'],
            )

    # process all datasets
    for dataset in datasets:
        run_pipeline_myelin2D(datadir, dataset, options, parameters)


def run_pipeline_myelin2D(datadir, dataset, options, parameters):
    """Run pipeline to segment myelinated axons in 2D.

    This function expects ... [TODO]
    """

    h5path = os.path.join(datadir, dataset + '.h5')

    # convert to h5
    if options['tif_to_h5']:
        utils.tif2h5_3D(datadir, dataset, 'data',
                        parameters['ds']['elsize'][:3],
                        parameters['ds']['axlab'][:3],
                        parameters['ds']['datapostfix'])
        utils.tif2h5_4D(datadir, dataset, 'probs',
                        parameters['ds']['elsize'],
                        parameters['ds']['axlab'],
                        parameters['ds']['probspostfix'])

    if options['run_eed']:
        # TODO: optional EED in matlab with subprocess
        # correct matlab's h5 probs_eed2 files
        utils.split_and_permute(h5path, 'probs_eed2',
                                ['probs0_eed2', 'probs1_eed2'])
        utils.copy_attributes(h5path, 'data',
                              ['probs0_eed2', 'probs1_eed2'])

    # data mask
    if options['datamask']:
        try:
            dset = 'maskDS'
            niifile = dataset + parameters['ds']['maskpostfix']
            utils.nii2h5(
                os.path.join(datadir, niifile),
                os.path.join(h5path, dset),
                inlayout='xyz', outlayout='zyx',
                )
        except:
            prob2mask(
                h5path_probs=os.path.join(h5path, 'probs'),
                h5path_out=os.path.join(h5path, 'maskDS'),
                channel=0,
                lower_threshold=parameters['ds']['threshold'],
                )
            outfilename = dataset + '_data.nii.gz'
            stack2stack(
                inputfile=os.path.join(h5path, 'data'),
                outputfile=os.path.join(datadir, outfilename),
                )

    # myelin mask
    if options['myelinmask']:
        try:
            dset = 'maskMM'
            niifile = dataset + parameters['ds']['maskpostfix']
            utils.nii2h5(
                os.path.join(datadir, niifile),
                os.path.join(h5path, dset),
                inlayout='xyz', outlayout='zyx',
                )
        except:
            prob2mask(
                h5path_probs=os.path.join(h5path, 'probs'),
                h5path_mask=os.path.join(h5path, 'maskDS'),
                h5path_out=os.path.join(h5path, 'maskMM'),
                channel=0,
                lower_threshold=parameters['mm']['lower_threshold'],
                upper_threshold=1.1,
                size=parameters['mm']['min_size'],
                )
    if options['myelinmask_multi']:
        prob2mask_levels(h5path, step=0.1,
                         min_size=parameters['mm']['min_size'])

    # connected components
    if options['connected_components']:
        CC_2D(
            h5path_in=os.path.join(h5path, 'maskMM'),
            h5path_mask=os.path.join(h5path, 'maskDS'),
            h5path_out=os.path.join(h5path, 'CC_2D'),
            )
    if options['connected_components_filter']:
        CC_2Dfilter(
            h5path_labels=os.path.join(h5path, 'CC_2D'),
            map_propnames=parameters['cc']['map_propnames'],
            criteria=parameters['cc']['criteria'],
            outputfile=os.path.join(datadir, dataset + '_fw.npy'),
            )
    if options['connected_components_mapping']:
        CC_2Dprops(
            h5path_labels=os.path.join(h5path, 'CC_2D'),
            basename=os.path.join(datadir, dataset) + '_fw',
            map_propnames=parameters['cc']['map_propnames'],
            h5path_out=os.path.join(h5path, 'CC_2D_props'),
            )

    # separate_sheaths
    if options['separate_sheaths']:
        separate_sheaths(
            h5path_in=os.path.join(h5path, 'CC_2D_props/label'),
            h5path_mmm=os.path.join(h5path, 'maskMM'),
            h5path_mask=os.path.join(h5path, 'maskDS'),
            h5path_out=os.path.join(h5path, 'labelMM'),
            MAdilation=parameters['ss']['MAdilation'],
            save_steps=True,
            )
    if options['separate_sheaths_weighted']:
        separate_sheaths(
            h5path_in=os.path.join(h5path, 'CC_2D_props/label'),
            h5path_mmm=os.path.join(h5path, 'maskMM'),
            h5path_mask=os.path.join(h5path, 'maskDS'),
            h5path_out=os.path.join(h5path, 'labelMM_sw'),
            h5path_lmm=os.path.join(h5path, 'labelMM'),
            h5path_wsmask=os.path.join(h5path, 'labelMM_steps/wsmask'),
            sigmoidweighting=parameters['ss']['sigmoidweighting'],
            save_steps=True,
            )

    # stats
    if options['seg_stats']:
        seg_stats(
            labelMA=os.path.join(h5path, 'CC_2D_props/label'),
            labelMF=os.path.join(h5path, 'labelMM_sw'),
            stats=['area', 'AD', 'centroid', 'eccentricity', 'solidity']
            )

    # convert to nifti
    # TODO: error handling if h5 not found
    # or use f.visititems(), but be careful not to overwrite manual steps
    if options['h5_to_nii']:

        if parameters['cn']['xXyY'] is not None:
            (x, X, y, Y) = parameters['cn']['xXyY']
            postfix = "_{:05d}-{:05d}_{:05d}-{:05d}".format(x, X, y, Y)
            dataset_cu = "{}{}".format(dataset, postfix)
            niftidir = os.path.join(datadir, dataset_cu)
            if not os.path.isdir(niftidir):
                os.makedirs(niftidir)
            for dset in parameters['cn']['dsets']:
                niidset = dset.replace('/', '_')
                outfilename = '{}_{}.nii.gz'.format(dataset_cu, niidset)
                stack2stack(
                    inputfile=os.path.join(h5path, dset),
                    outputfile=os.path.join(niftidir, outfilename),
                    xyzct=(x, X, y, Y, 0, 0, 0, 0, 0, 0)
                    )

        else:
            for dset in parameters['cn']['dsets']:
                niidset = dset.replace('/', '_')
                outfilename = '{}_{}.nii.gz'.format(dataset, niidset)
                stack2stack(
                    inputfile=os.path.join(h5path, dset),
                    outputfile=os.path.join(datadir, outfilename),
                    )


def prob2mask_levels(h5path, step=0.1, min_size=1000):
    """Threshold a probability maps with a range of thresholds."""

    def frange(x, y, jump):
        """Range for floats."""
        while x < y:
            yield x
            x += jump

    for thr in frange(step, 1.0, step):
        prob2mask(
            os.path.join(h5path, 'probs'),
            h5path_out=os.path.join(h5path, 'maskMMlevels/{:.2f}'.format(thr)),
            channel=0,
            lower_threshold=thr, upper_threshold=1.1,
            size=min_size
            )
