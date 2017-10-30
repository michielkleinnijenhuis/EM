#!/usr/bin/env python

"""Parse arguments for functions in the wmem package.

"""


def parse_common(parser):

    parser.add_argument(
        '-D', '--dataslices',
        nargs='*',
        type=int,
        help="""
        Data slices, specified as triplets of <start> <stop> <step>;
        setting any <stop> to 0 or will select the full extent;
        provide triplets in the order of the input datset.
        """
        )

    parser.add_argument(
        '-M', '--usempi',
        action='store_true',
        help='use mpi4py'
        )

    parser.add_argument(
        '-S', '--save_steps',
        action='store_true',
        help='save intermediate results'
        )

    parser.add_argument(
        '-P', '--protective',
        action='store_true',
        help='protect against overwriting data'
        )

    return parser


def parse_downsample_slices(parser):

    parser.add_argument(
        'inputdir',
        help='a directory with images'
        )
    parser.add_argument(
        'outputdir',
        help='the output directory'
        )

    parser.add_argument(
        '-r', '--regex',
        default='*.tif',
        help='regular expression to select files'
        )
    parser.add_argument(
        '-f', '--downsample_factor',
        type=int,
        default=4,
        help='the factor to downsample the images by'
        )

    return parser


def parse_series2stack(parser):

    parser.add_argument(
        'inputdir',
        help='a directory with images'
        )
    parser.add_argument(
        'outputfile',
        help='the path to the output dataset'
        )

    parser.add_argument(
        '-r', '--regex',
        default='*.tif',
        help='regular expression to select files with'
        )
    parser.add_argument(
        '-d', '--datatype',
        default='',
        help='the numpy-style output datatype'
        )
    parser.add_argument(
        '-o', '--outlayout',
        default='zyx',
        help='axis labels: any permutation of "zyx"'
        )
    parser.add_argument(
        '-e', '--element_size_um',
        nargs=3,
        type=float,
        default=[None, None, None],
        help='dataset element sizes in the order of outlayout'
        )
    parser.add_argument(
        '-s', '--chunksize',
        type=int,
        nargs=3,
        default=[20, 20, 20],
        help='hdf5 chunk sizes in the order of outlayout'
        )

    return parser


def parse_stack2stack(parser):

    parser.add_argument(
        'inputfile',
        help='the inputfile'
        )
    parser.add_argument(
        'outputfile',
        help='the outputfile'
        )

    parser.add_argument(
        '-x',
        type=int,
        default=0,
        help='lower x-index'
        )
    parser.add_argument(
        '-X',
        type=int,
        default=0,
        help='upper x-index'
        )
    parser.add_argument(
        '-y',
        type=int,
        default=0,
        help='lower y-index'
        )
    parser.add_argument(
        '-Y',
        type=int,
        default=0,
        help='upper y-index'
        )
    parser.add_argument(
        '-z',
        type=int,
        default=0,
        help='lower z-index'
        )
    parser.add_argument(
        '-Z',
        type=int,
        default=0,
        help='upper z-index'
        )
    parser.add_argument(
        '-c',
        type=int,
        default=0,
        help='lower c-index'
        )
    parser.add_argument(
        '-C',
        type=int,
        default=0,
        help='upper C-index'
        )
    parser.add_argument(
        '-t',
        type=int,
        default=0,
        help='lower t-index'
        )
    parser.add_argument(
        '-T',
        type=int,
        default=0,
        help='upper T-index'
        )

    parser.add_argument(
        '-s', '--chunksize',
        type=int,
        nargs='*',
        help='hdf5 chunk sizes (in order of outlayout)'
        )
    parser.add_argument(
        '-e', '--element_size_um',
        type=float,
        nargs='*',
        help='dataset element sizes (in order of outlayout)'
        )

    parser.add_argument(
        '-i', '--inlayout',
        default='',
        help='the data layout of the input'
        )
    parser.add_argument(
        '-o', '--outlayout',
        default='',
        help='the data layout for output'
        )

    parser.add_argument(
        '-p', '--dset_name',
        default='',
        help='the identifier of the datablock'
        )
    parser.add_argument(
        '-b', '--blockoffset',
        type=int,
        nargs='*',
        help='...'
        )

    parser.add_argument(
        '-a', '--additional_outputs',
        nargs='*',
        default=[],
        help='output the stack in jpg, png, tif, nii, h5'
        )
    parser.add_argument(
        '-n', '--nzfills',
        type=int,
        default=5,
        help='number of characters at the end that define z'
        )

    parser.add_argument(
        '-r', '--downscale',
        type=int,
        nargs='*',
        default=[],
        help='factors to downscale (in order of outlayout)'
        )
    parser.add_argument(
        '-d', '--datatype',
        default='',
        help='the numpy-style output datatype'
        )
    parser.add_argument(
        '-u', '--uint8conv',
        action='store_true',
        help='convert data to uint8'
        )

    return parser


def parse_prob2mask(parser):

    parser.add_argument(
        'inputfile',
        help='the path to the input dataset'
        )
    parser.add_argument(
        'outputfile',
        default='',
        help='the path to the output dataset'
        )

    parser.add_argument(
        '-c', '--channel',
        type=int,
        help='pick a channel from a 4D (zyx)c dataset'
        )

    parser.add_argument(
        '-i', '--inputmask',
        nargs=2,
        default=None,
        help='additional mask to apply to the output'
        )

    parser.add_argument(
        '-b', '--blockreduce',
        nargs=3,
        type=int,
        default=None,
        help='downsample the output with scikit-image block_reduce (np.amax)'
        )

    parser.add_argument(
        '-l', '--lower_threshold',
        type=float,
        default=0,
        help='the lower threshold to apply to the dataset'
        )
    parser.add_argument(
        '-u', '--upper_threshold',
        type=float,
        default=1,
        help='the lower threshold to apply to the dataset'
        )
    parser.add_argument(
        '-j', '--step',
        type=float,
        default=1.0,
        help='multiple lower thresholds between 0 and 1 using this step'
        )
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=0,
        help='remove any components smaller than this number of voxels'
        )
    parser.add_argument(
        '-d', '--dilation',
        type=int,
        default=0,
        help='perform a mask dilation with a ball-shaped selem of this size'
        )

    return parser


def parse_mergeblocks(parser):

    parser.add_argument(
        'inputfiles',
        nargs='*',
        help="""paths to hdf5 datasets <filepath>.h5/<...>/<dataset>:
                datasets to merge together"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                merged dataset"""
        )

    parser.add_argument(
        '-b', '--blockoffset',
        nargs=3,
        type=int,
        default=[0, 0, 0],
        help='offset of the datablock'
        )
    parser.add_argument(
        '-p', '--blocksize',
        nargs=3,
        type=int,
        default=[0, 0, 0],
        help='size of the datablock'
        )
    parser.add_argument(
        '-q', '--margin',
        nargs=3,
        type=int,
        default=[0, 0, 0],
        help='the datablock overlap used'
        )
    parser.add_argument(
        '-s', '--fullsize',
        nargs=3,
        type=int,
        default=[],
        help='the size of the full dataset'
        )

    parser.add_argument(
        '-l', '--is_labelimage',
        action='store_true',
        help='flag to indicate labelimage'
        )
    parser.add_argument(
        '-r', '--relabel',
        action='store_true',
        help='apply incremental labeling to each block'
        )
    parser.add_argument(
        '-n', '--neighbourmerge',
        action='store_true',
        help='merge overlapping labels'
        )
    parser.add_argument(
        '-F', '--save_fwmap',
        action='store_true',
        help='save the forward map (.npy)'
        )

    parser.add_argument(
        '-B', '--blockreduce',
        nargs=3,
        type=int,
        default=[],
        help='downsample the datablocks'
        )
    parser.add_argument(
        '-f', '--func',
        default='np.amax',
        help='function used for downsampling'
        )

    return parser


def parse_downsample_blockwise(parser):

    parser.add_argument(
        'inputfile',
        help='the path to the input dataset'
        )
    parser.add_argument(
        'outputfile',
        help='the path to the output dataset'
        )

    parser.add_argument(
        '-B', '--blockreduce',
        nargs=3,
        type=int,
        default=[1, 7, 7],
        help='the blocksize'
        )
    parser.add_argument(
        '-f', '--func',
        default='np.amax',
        help='the function to use for blockwise reduction'
        )

    return parser


def parse_connected_components(parser):

    parser.add_argument(
        'inputfile',
        help='the path to the input dataset'
        )
    parser.add_argument(
        'outputfile',
        default='',
        help='the path to the output dataset'
        )

    parser.add_argument(
        '-m', '--mode',
        help='...'
        )

    parser.add_argument(
        '-b', '--basename',
        default='',
        help='...'
        )

    parser.add_argument(
        '--maskDS',
        default='',
        help='...'
        )
    parser.add_argument(
        '--maskMM',
        default='',
        help='...'
        )
    parser.add_argument(
        '--maskMB',
        default='',
        help='...'
        )

    parser.add_argument(
        '-d', '--slicedim',
        type=int,
        default=0,
        help='...'
        )

    parser.add_argument(
        '-p', '--map_propnames',
        nargs='*',
        help='...'
        )

    parser.add_argument(
        '-q', '--min_size_maskMM',
        type=int,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-a', '--min_area',
        type=int,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-A', '--max_area',
        type=int,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-I', '--max_intensity_mb',
        type=float,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-E', '--max_eccentricity',
        type=float,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-e', '--min_euler_number',
        type=float,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-s', '--min_solidity',
        type=float,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-n', '--min_extent',
        type=float,
        default=None,
        help='...'
        )

    return parser


def parse_merge_slicelabels(parser):

    parser.add_argument(
        'inputfile',
        help='the path to the input dataset'
        )
    parser.add_argument(
        'outputfile',
        default='',
        help='the path to the output dataset'
        )

    parser.add_argument(
        '--maskMM',
        default='',
        help='...'
        )
    parser.add_argument(
        '-d', '--slicedim',
        type=int,
        default=0,
        help='...'
        )

    parser.add_argument(
        '-m', '--mode',
        help='...'
        )

    parser.add_argument(
        '-q', '--offsets',
        type=int,
        default=2,
        help='...'
        )
    parser.add_argument(
        '-o', '--threshold_overlap',
        type=float,
        default=None,
        help='...'
        )

    parser.add_argument(
        '-s', '--min_labelsize',
        type=int,
        default=0,
        help='...'
        )
    parser.add_argument(
        '-l', '--close',
        nargs='*',
        type=int,
        default=None,
        help='...'
        )
    parser.add_argument(
        '-r', '--relabel',
        action='store_true',
        help='...'
        )

    return parser


def parse_fill_holes(parser):

    parser.add_argument(
        'inputfile',
        help='the path to the input dataset'
        )
    parser.add_argument(
        'outputfile',
        default='',
        help='the path to the output dataset'
        )

    parser.add_argument(
        '-m', '--methods',
        default="2",
        help='method(s) used for filling holes'
        )  # TODO: document methods
    parser.add_argument(
        '-s', '--selem',
        nargs='*',
        type=int,
        default=[3, 3, 3],
        help='the structuring element used in methods 1, 2 & 3'
        )

    parser.add_argument(
        '-l', '--labelmask',
        default='',
        help='the path to a mask: labelvolume[~labelmask] = 0'
        )

    parser.add_argument(
        '--maskDS',
        default='',
        help='the path to a dataset mask'
        )
    parser.add_argument(
        '--maskMM',
        default='',
        help='the path to a mask of the myelin compartment'
        )
    parser.add_argument(
        '--maskMX',
        default='',
        help='the path to a mask of a low-thresholded myelin compartment'
        )

    parser.add_argument(
        '--outputholes',
        default='',
        help='the path to the output dataset (filled holes)'
        )
    parser.add_argument(
        '--outputMA',
        default='',
        help='the path to the output dataset (updated myel. axon mask)'
        )  # TODO: then need to update labelvolume as well?!
    parser.add_argument(
        '--outputMM',
        default='',
        help='the path to the output dataset (updated myelin mask)'
        )

    return parser


def parse_separate_sheaths(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with myelinated axon labels
                used as seeds for watershed dilated with scipy's
                <grey_dilation(<dataset>, size=[3, 3, 3])>"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with separated myelin sheaths"""
        )

    parser.add_argument(
        '--maskWS',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                maskvolume of the myelin space
                to which the watershed will be constrained"""
        )
    parser.add_argument(
        '--maskDS',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                maskvolume of the data"""
        )
    parser.add_argument(
        '--maskMM',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                maskvolume of the myelin compartment"""
        )
    parser.add_argument(
        '-d', '--dilation_iterations',
        type=int,
        default=0,
        help="""number of iterations for binary dilation
                of the myelinated axon compartment
                (as derived from inputfile):
                it determines the maximal extent
                the myelin sheath can be from the axon"""
        )

    parser.add_argument(
        '--distance',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                distance map volume used in for watershed"""
        )
    parser.add_argument(
        '--labelMM',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume used in calculating the median sheath thickness
                of each sheath for the sigmoid-weighted distance map"""
        )
    parser.add_argument(
        '-w', '--sigmoidweighting',
        type=float,
        help="""the steepness of the sigmoid
                <scipy.special.expit(w * np.median(sheath_thickness)>"""
        )
    parser.add_argument(
        '-m', '--margin',
        type=int,
        default=50,
        help="""margin of the box used when calculating
                the sigmoid-weighted distance map"""
        )

    return parser


def parse_agglo_from_labelsets(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume of oversegmented supervoxels"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with agglomerated labels"""
        )

    parser.add_argument(
        '-l', '--labelset_files',
        nargs='*',
        default=[],
        help="""files with label mappings, either
                A.) pickled python dictionary
                {label_new1: [<label_1>, <label_2>, <...>],
                 label_new2: [<label_6>, <label_3>, <...>]}
                B.) ascii files with on each line:
                <label_new1>: <label_1> <label_2> <...>
                <label_new2>: <label_6> <label_3> <...>"""
        )
    parser.add_argument(
        '-f', '--fwmap',
        default=[],
        help="""numpy .npy file with vector of length np.amax(<labels>) + 1
                representing the forward map to apply, e.g.
                fwmap = np.array([0, 0, 4, 8, 8]) will map
                the values 0, 1, 2, 3, 4 in the labelvolume
                to the new values 0, 0, 4, 8, 8;
                i.e. it will delete label 1, relabel label 2 to 4,
                and merge labels 3 & 4 to new label value 8
                """
        )

    return parser


def parse_watershed_ics(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                input to the watershed algorithm"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with watershed oversegmentation"""
        )

    parser.add_argument(
        '--masks',
        nargs='*',
        default=[],
        help="""string of paths to hdf5 datasets <filepath>.h5/<...>/<dataset>
                and logical functions (NOT, AND, OR, XOR)
                to add the hdf5 masks together
                (NOT operates on the following dataset), e.g.
                NOT <f1>.h5/<m1> AND <f1>.h5/<m2> will first invert m1,
                then combine the result with m2 through the AND operator
                starting point is np.ones(<in>.shape[:3], dtype='bool')"""
        )

    parser.add_argument(
        '--seedimage',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with seeds to the watershed"""
        )
    parser.add_argument(
        '-l', '--lower_threshold',
        type=float,
        default=None,
        help='the lower threshold for generating seeds from the dataset'
        )
    parser.add_argument(
        '-u', '--upper_threshold',
        type=float,
        default=None,
        help='the upper threshold for generating seeds from the dataset'
        )
    parser.add_argument(
        '-s', '--seed_size',
        type=int,
        default=64,
        help='the minimal size of a seed label'
        )

    return parser


def parse_agglo_from_labelmask(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume of oversegmented supervoxels"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with agglomerated labels"""
        )

    parser.add_argument(
        '-l', '--labelvolume',
        default='',
        help='...'
        )
    parser.add_argument(
        '-r', '--ratio_threshold',
        type=float,
        default=0,
        help='...'
        )

    return parser


def parse_remap_labels(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume"""
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume with deleted/merged/split labels"""
        )

    parser.add_argument(
        '-D', '--delete_labels',
        nargs='*',
        type=int,
        default=[],
        help='list of labels to delete'
        )
    parser.add_argument(
        '-d', '--delete_files',
        nargs='*',
        default=[],
        help='list of files with labelsets to delete'
        )
    parser.add_argument(
        '--except_files',
        nargs='*',
        default=[],
        help='...'
        )
    parser.add_argument(
        '-E', '--merge_labels',
        nargs='*',
        type=int,
        default=[],
        help='list with pairs of labels to merge'
        )
    parser.add_argument(
        '-e', '--merge_files',
        nargs='*',
        default=[],
        help='list of files with labelsets to merge'
        )
    parser.add_argument(
        '-F', '--split_labels',
        nargs='*',
        type=int,
        default=[],
        help='list of labels to split'
        )
    parser.add_argument(
        '-f', '--split_files',
        nargs='*',
        default=[],
        help='list of files with labelsets to split'
        )
    parser.add_argument(
        '-A', '--aux_labelvolume',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                labelvolume from which to take alternative labels"""
        )
    parser.add_argument(
        '-q', '--min_labelsize',
        type=int,
        default=0,
        help='the minimum size of the labels'
        )
    parser.add_argument(
        '-p', '--min_segmentsize',
        type=int,
        default=0,
        help='the minimum segment size of non-contiguous labels'
        )
    parser.add_argument(
        '-k', '--keep_only_largest',
        action='store_true',
        help='keep only the largest segment of a split label'
        )
    parser.add_argument(
        '-O', '--conncomp',
        action='store_true',
        help='relabel split labels with connected component labeling'
        )
    parser.add_argument(
        '-n', '--nifti_output',
        action='store_true',
        help='also write output to nifti'
        )
    parser.add_argument(
        '-m', '--nifti_transpose',
        action='store_true',
        help='transpose the output before writing to nifti'
        )

    return parser


def parse_nodes_of_ranvier(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    parser.add_argument(
        '-s', '--min_labelsize',
        type=int,
        default=0,
        help='...'
        )
    parser.add_argument(
        '-R', '--remove_small_labels',
        action='store_true',
        help='remove the small labels before further processing'
        )

    parser.add_argument(
        '--boundarymask',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    parser.add_argument(
        '-m', '--merge_methods',
        nargs='*',
        default=['neighbours'],
        help='neighbours, conncomp, and/or watershed'
        )
    parser.add_argument(
        '-o', '--overlap_threshold',
        type=int,
        default=20,
        help='for neighbours'
        )

    parser.add_argument(
        '-r', '--searchradius',
        nargs=3,
        type=int,
        default=[100, 30, 30],
        help='for watershed'
        )
    parser.add_argument(
        '--data',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )
    parser.add_argument(
        '--maskMM',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    return parser


def parse_filter_NoR(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    parser.add_argument(
        '--input2D',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    return parser


def parse_convert_dm3(parser):

    parser.add_argument(
        'inputfile',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )
    parser.add_argument(
        'outputfile',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    parser.add_argument(
        '--input2D',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    return parser


def parse_seg_stats(parser):

    parser.add_argument(
        '--h5path_labelMA',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )
    parser.add_argument(
        '--h5path_labelMF',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )
    parser.add_argument(
        '--h5path_labelUA',
        default='',
        help="""path to hdf5 dataset <filepath>.h5/<...>/<dataset>:
                """
        )

    parser.add_argument(
        '--stats',
        nargs='*',
        default=['area', 'AD', 'centroid', 'eccentricity', 'solidity'],
        help="""the statistics to export"""
        )

    parser.add_argument(
        '--outputbasename',
        help="""basename for the output"""
        )

    return parser