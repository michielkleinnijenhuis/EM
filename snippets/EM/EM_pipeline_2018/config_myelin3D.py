def config(parameters={}, run_from='', run_upto='', run_only=''):

    # dataset
    datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_00'
    datasets = ['T4_1']

    steps = [
        'normalize_datasets', 'dm3_to_tif', 'register', 'downsample', 'tif_to_h5',
        'datamask', 'myelinmask', 'myelinmask_multi',
        'connected_components', 'connected_components_filter', 'connected_components_mapping',
        'separate_sheaths', 'separate_sheaths_weighted',
        'seg_stats', 'h5_to_nii',
        ]

    # options
    options = {
        'normalize_datasets': False,
        'dm3_to_tif': True,
        'register': True,
        'downsample': True,
        'tif_to_h5': True,
        'run_eed': True,
        'datamask': True,
        'myelinmask': True,
        'myelinmask_multi': False,
        'connected_components': True,
        'connected_components_filter': True,
        'connected_components_mapping': True,
        'separate_sheaths': True,
        'separate_sheaths_weighted': True,
        'seg_stats': True,
        'h5_to_nii': True,
        }

    # if run_from:
    #     idx = steps.index(run_from)
    #     for i, step in enumerate(steps):
    #         options[step] = i >= idx
    if run_from:
        idx = steps.index(run_from)
        for step in steps[:idx]:
            options[step] = False
    if run_upto:
        idx = steps.index(run_upto)
        for step in steps[idx:]:
            options[step] = False
    if run_only:
        for step in steps:
            options[step] = run_only == step

    # dataset parameters
    parameters['ds'] = {
        'dm3dir':
        'elsize': [1, 1, 1, 1],
        'axlab': ['z', 'y', 'x', 'c'],
        'datapostfix': '_norm.tif',
        'probspostfix': '_norm_probs.tif',
        'maskpostfix': '_maskDS_manual.nii.gz',
        'threshold': -1,
        }

    # myelin mask parameters
    parameters['mm'] = {
        'lower_threshold': 0.5,
        'min_size': 1000,
        'maskpostfix': '_maskMM_manual.nii.gz',
        }

    # connected component parameters
    parameters['cc'] = {
        'map_propnames': [
            'label',
            'area',
            'eccentricity',
            'euler_number',
            'extent',
            'solidity',
            ],
        'criteria': (1000, 100000, None, 1.0, 0.00, 0, 0.00),
        # FIXME: make criteria into dictionary
        # (min_area,
        #  max_area,
        #  max_intensity_mb,
        #  max_eccentricity,
        #  min_solidity,
        #  min_euler_number,
        #  min_extent) = criteria
        }

    # separate sheaths parameters
    parameters['ss'] = {
        'MAdilation': 100,
        'sigmoidweighting': 0.01,
        }

    # convert-to-nifti parameters
    parameters['cn'] = {
        'xXyY': (1000, 4000, 1000, 4000),  # None
        'dsets': [
            'data',
            'probs',
            'maskDS',
            'maskMM',
            'CC_2D',
            'CC_2D_props/label',
            'CC_2D_props/area',
            'CC_2D_props/eccentricity',
            'CC_2D_props/euler_number',
            'CC_2D_props/extent',
            'CC_2D_props/solidity',
            # 'CC_2D_props/label_remapped',
            'labelMM',
            'labelMM_steps/wsmask',
            'labelMM_steps/distance_simple',
            'labelMM_sw',
            ],
        }

    return datadir, datasets, options, parameters
