#!/bin/bash


function prep_environment {
    # Set paths, define aliases, load modules and build executables.

    local scriptdir=$1
    local compute_env=$2

    # for easy rsyncing between systems
    host_jal=michielk@jalapeno.fmrib.ox.ac.uk
    host_arc=ndcn0180@arcus-b.arc.ox.ac.uk
    host_pmc=mkleinnijenhuis@processed.pmc_research.op.umcutrecht.nl
    host_hpc=mkleinnijenhuis@hpct01.op.umcutrecht.nl
    scriptdir_loc="$HOME/workspace/EM"
    scriptdir_jal='/home/fs0/michielk/workspace/EM'
    scriptdir_arc='/home/ndcn0180/workspace/EM'
    scriptdir_pmc='/home/pmc_research/mkleinnijenhuis/workspace/EM'
    scriptdir_hpc='/home/pmc_research/mkleinnijenhuis/workspace/EM'

    if [ "$compute_env" == "ARCB" ]
    then
        PATH="$DATA/anaconda2/bin:$PATH"
        HDF5_PREFIX='/data/ndcn-fmrib-water-brain/ndcn0180/workspace/hdf5'
        PATH="$HDF5_PREFIX/bin:$PATH"
        CONDA_PATH="$( conda info --root )"
        # PYTHONPATH="$scriptdir"
        PYTHONPATH="$PYTHONPATH:$HOME/workspace/pyDM3reader"
        imagej='/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64'
        ilastik="$HOME/workspace/ilastik-1.2.2post1-Linux/run_ilastik.sh"

        # module load hdf5-parallel/1.8.17_mvapich2_gcc
        module load matlab/R2015a

        eed_tgtdir="$scriptdir/bin"
        eed_tbxdir="$HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b"
        eed_script="$scriptdir/wmem/EM_eed_simple.m"
        [ ! -f "$eed_tgtdir/EM_eed_simple" ] &&
            deployed_eed "$eed_tgtdir" "$eed_tbxdir" "$eed_script"

    elif [ "$compute_env" == "ARC" ]
    then

        module load python/2.7 mpi4py/1.3.1 hdf5-parallel/1.8.14_mvapich2
        module load matlab/R2015a

        PATH="$DATA/anaconda2/bin:$PATH"
        CONDA_PATH="$( conda info --root )"
        # PYTHONPATH="$scriptdir:$PYTHONPATH"
        PYTHONPATH="$HOME/workspace/pyDM3reader:$PYTHONPATH"
        imagej='/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64'
        ilastik="$HOME/workspace/ilastik-1.2.2post1-Linux/run_ilastik.sh"

        eed_tgtdir="$scriptdir/bin"
        eed_tbxdir="$HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b"
        eed_script="$scriptdir/wmem/EM_eed_simple.m"
        [ ! -f "$eed_tgtdir/EM_eed_simple" ] &&
            deployed_eed "$eed_tgtdir" "$eed_tbxdir" "$eed_script"

    elif [ "$compute_env" == "JAL" ]
    then

        DATA='/vols/Data/km/michielk/oxdata/P01'
        PATH="/vols/Data/km/michielk/anaconda2/bin:$PATH"
        HDF5_PREFIX='/vols/Data/km/michielk/workspace/hdf5'
        PATH="$HDF5_PREFIX/bin:$PATH"
        CONDA_PATH="$( conda info --root )"
        # PYTHONPATH="$scriptdir:$PYTHONPATH"
        PYTHONPATH="$HOME/workspace/pyDM3reader:$PYTHONPATH"
        # imagej=/opt/fmrib/ImageJ  # TODO
        ilastik='/vols/Data/km/michielk/ilastik-1.2.2post1-Linux/run_ilastik.sh'

        eed_tgtdir="$scriptdir/bin"
        eed_tbxdir="$HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b"
        eed_script="$scriptdir/wmem/EM_eed_simple.m"
        [ ! -f "$eed_tgtdir/EM_eed_simple" ] &&
            deployed_eed "$eed_tgtdir" "$eed_tbxdir" "$eed_script"

    elif [ "$compute_env" == "LOCAL" ]
    then
        DATA="$HOME/oxdata/P01"
        PATH="$HOME/anaconda2/bin:$PATH"
        CONDA_PATH="$( conda info --root )"
        PYTHONPATH="$scriptdir"
        PYTHONPATH="$PYTHONPATH:$HOME/workspace/pyDM3reader"
        imagej='/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
        ilastik='/Applications/ilastik-1.2.2post1-OSX.app/Contents/MacOS/ilastik'

        eed_tgtdir="$scriptdir/bin"
        eed_tbxdir="$HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b"
        eed_script="$scriptdir/wmem/EM_eed_simple.m"
        [ ! -f "$eed_tgtdir/EM_eed_simple.app/Contents/MacOS/EM_eed_simple" ] &&
            deployed_eed "$eed_tgtdir" "$eed_tbxdir" "$eed_script"

    elif [ "$compute_env" == "RIOS013" ]
    then
        DATA="$HOME/oxdata/P01"
        PATH="$HOME/anaconda2/bin:$PATH"
        CONDA_PATH="$( conda info --root )"
        PYTHONPATH="$scriptdir"
        PYTHONPATH="$PYTHONPATH:$HOME/workspace/pyDM3reader"
        imagej='/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
        ilastik='/Applications/ilastik-1.2.2post1-OSX.app/Contents/MacOS/ilastik'

        eed_tgtdir="$scriptdir/bin"
        eed_tbxdir="$HOME/workspace/matlab/toolboxes/coherencefilter_version5b"
        eed_script="$scriptdir/wmem/EM_eed_simple.m"
        [ ! -f "$eed_tgtdir/EM_eed_simple.app/Contents/MacOS/EM_eed_simple" ] &&
            deployed_eed "$eed_tgtdir" "$eed_tbxdir" "$eed_script"

    elif [ "$compute_env" == "HPC" ]
    then
        DATA="/hpc/pmc_rios/mkleinnijenhuis/oxdata/P01"
        PATH="/hpc/local/CentOS7/pmc_rios/anaconda3/bin:$PATH"
        CONDA_PATH="$( conda info --root )"
        PYTHONPATH="$scriptdir"
        PYTHONPATH="$PYTHONPATH:$HOME/workspace/pyDM3reader"
        imagej='/hpc/local/CentOS7/pmc_rios/workspace/Fiji.app/ImageJ-linux64'
        ilastik='/hpc/local/CentOS7/pmc_rios/workspace/ilastik-1.3.2post1-Linux/run_ilastik.sh'

        eed_tgtdir="$scriptdir/bin"
        eed_tbxdir="$HOME/workspace/matlab/toolboxes/coherencefilter_version5b"
        eed_script="$scriptdir/wmem/EM_eed_simple.m"
        [ ! -f "$eed_tgtdir/EM_eed_simple" ] &&
            deployed_eed "$eed_tgtdir" "$eed_tbxdir" "$eed_script"

    fi

}


function prep_dataset {
    # Load the parameters of a dataset and cd to its directory.

    local dataset=$1
    local bs=$2  # blocksize

    # load the dataset parameters
    dataset_parameters $dataset

    # prepare directory
    datadir=$basedir/$dataset &&
        mkdir -p $datadir &&
            cd $datadir

    # prepare the block directory
    blockdir=$datadir/blocks_$bs &&
        mkdir -p $blockdir &&
            datastems_blocks

    # Set the identifier of the in-plane downsampled dataset
    dataset_ds="$dataset$dspf$ds"

}


function datastems_blocks {
    # Generate an array <datastems> of block identifiers.
    # taking the form "dataset_x-X_y-Y_z-Z"
    # with voxel coordinates zero-padded to 5 digits

    local verbose=$1
    local x X y Y z Z
    local dstem

    unset datastems
    datastems=()

    for x in `seq 0 $xs $(( xmax-1 ))`; do
        X=$( get_coords_upper $x $xm $xs $xmax)
        x=$( get_coords_lower $x $xm )
        for y in `seq 0 $ys $(( ymax-1 ))`; do
            Y=$( get_coords_upper $y $ym $ys $ymax)
            y=$( get_coords_lower $y $ym )
            for z in `seq 0 $zs $(( zmax-1 ))`; do
                Z=$( get_coords_upper $z $zm $zs $zmax)
                z=$( get_coords_lower $z $zm )

                dstem="$( get_datastem $dataset $x $X $y $Y $z $Z )"
                datastems+=( "$dstem" )
                if [ "$verbose" == "-v" ]; then
                    echo "$dstem"
                fi

            done
        done
    done

}


function datastem2coords {
    # Extract start-stop indices from blockname.

    CO=${datastem#"$dataset"}
    x=${CO:1:5}; x=$(strip_leading_zeroes $x);
    X=${CO:7:5}; X=$(strip_leading_zeroes $X);
    y=${CO:13:5}; y=$(strip_leading_zeroes $y);
    Y=${CO:19:5}; Y=$(strip_leading_zeroes $Y);
    z=${CO:25:5}; z=$(strip_leading_zeroes $z);
    Z=${CO:31:5}; Z=$(strip_leading_zeroes $Z);

    echo $x $X $y $Y $z $Z

}


function downsampled_coords {
    # Find downsampled (x,X,y,Y) coordinates from full coordinates.

    #echo "$(( $1 / ds)) $(( $2 / ds)) $(( $3 / ds)) $(( $4 / ds)) $5 $6"
    k=$(( ds - 1 ))
    echo "$(( ($1 + k) / ds)) $(( ($2 + k) / ds)) $(( ($3 + k) / ds)) $(( ($4 + k) / ds)) $5 $6"


}


function coordsXYZ_to_slicesZYX {
    # Convert coordinate set to python slices form.

    local start_x=$1
    local stop_x=$2
    local start_y=$3
    local stop_y=$4
    local start_z=$5
    local stop_z=$6

    k=$(( ds - 1 ))
    [[ "$stop_x" -eq "$(( ($xmax + k) / ds ))" ]] && stop_x=0
    [[ "$stop_y" -eq "$(( ($ymax + k) / ds ))" ]] && stop_y=0
    [[ "$stop_z" -eq "$zmax" ]] && stop_z=0

    echo "$start_z $stop_z 1 $start_y $stop_y 1 $start_x $stop_x"

}


function strip_leading_zeroes {
    # Strip leading zeroes from a number.

    local num=$1

    num=${num#"${num%%[!0]*}"}
    [[ ! -z "$num" ]] && echo "$num" || echo "0"

}


function get_coords_upper {
    # Get upper coordinate of the block.
    # Adds the blocksize and margin to lower coordinate,
    # and picks between that value and max extent of the dimension,
    # whichever is lower

    local co=$1
    local margin=$2
    local size=$3
    local comax=$4
    local CO

    CO=$(( co + size + margin )) &&
        CO=$(( CO < comax ? CO : comax ))

    echo "$CO"

}


function get_coords_lower {
    # Get lower coordinate of the block.
    # Subtracts the margin,
    # and picks between that value and 0,
    # whichever is higher

    local co=$1
    local margin=$2

    co=$(( co - margin )) &&
        co=$(( co > 0 ? co : 0 ))

    echo "$co"

}


function get_datastem {
    # Get a block identifier from coordinates.

    local dataset=$1
    local x=$2
    local X=$3
    local y=$4
    local Y=$5
    local z=$6
    local Z=$7

    local xrange=`printf %05d $x`-`printf %05d $X`
    local yrange=`printf %05d $y`-`printf %05d $Y`
    local zrange=`printf %05d $z`-`printf %05d $Z`

    local dstem=${dataset}_${xrange}_${yrange}_${zrange}

    echo "$dstem"

}


function get_datastem_index {
    # Locate a block identifier in an array and return it's index.

    local datastem=$1

    for i in "${!datastems[@]}"; do
       if [[ "${datastems[$i]}" = "${datastem}" ]]; then
           echo "${i}";
       fi
    done

}


function find_missing_datastems {
    # Filter datastems array to retain only blocks that are missing.
    # Searches filenames.

    local datadir=$1
    local postfix=$2
    local ext=$3

    unset missing
    declare -a missing

    for datastem in "${datastems[@]}"; do
        [ -f "$datadir$datastem$postfix.$ext" ] ||
            { missing+=( "$datastem" ); echo $datastem ; }
    done

    datastems=( "${missing[@]}" )

}


function find_missing_h5 {
    # Filter datastems array to retain only blocks that are missing.
    # Searches h5 datasets.

    local datadir=$1
    local postfix=$2
    local dset=$3
    local h5path
    local ret_code

    unset missing
    declare -a missing

    for datastem in ${datastems[@]}; do
        h5path="$datadir/$datastem$postfix.h5/$dset"
        h5ls "$h5path" > /dev/null 2> /dev/null
        ret_code=$?
        if [ "$ret_code" -ne "0" ]; then
            missing+=( "$datastem" )
            echo "$datastem"
        fi
    done

    datastems=( "${missing[@]}" )

}


function set_datastems {
    #

    [ ! -z "$stemsmode" ] &&
        datastems_blocks
    [ "$stemsmode" == "m" ] &&
        find_missing_h5 $datadir/blocks_$bs $opf $ods

}


function get_infiles {
    # Get infiles array of h5paths to all blocks with a postfix ipf.

    local f
    local blockformat='_?????-?????_?????-?????_?????-?????'
    local regex="$dataset$blockformat$ipf.h5"

    for f in `ls $datadir/blocks_$bs/$regex`; do
        infiles+=( "$f/$ids" )
    done

}


function get_infiles_datastems {
    # Get infiles array of h5paths to all blocks with a postfix ipf.

    for datastem in "${datastems[@]}"; do
        infiles+=( "$datadir/blocks_$bs/$datastem$ipf.h5/$ids" )
    done

}


function get_infiles_datastem_indices {
    # Get infiles array of h5paths to all blocks with a postfix ipf.

    local datastem_indices="$*"
    local ds_idx

    for datastem_index in $datastem_indices; do
        infiles+=( "$datadir/blocks_$bs/${datastems[datastem_index]}$ipf.h5/$ids" )
    done

}


function script_mpi_conda {
    # Write script. (deprecated?)

    local scriptfile=$1
    shift
    local cmd="$*"

    echo '#!/bin/bash' > $scriptfile
    echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
    echo "source activate root" >> $scriptfile
    echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile

    echo "$cmd"  >> $scriptfile

    chmod +x $scriptfile

}


function deployed_eed {
    # Build EED standalone.

    local eed_tgtdir=$1
    local eed_tbxdir=$2
    local eed_script=$3

    mkdir -p $eed_tgtdir && cd $eed_tgtdir

    mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh \
    -m $eed_script -a $eed_tbxdir

    cd $datadir

}


function get_cmd_dm3convert {
    # Get the command for converting dm3 files.

    echo python -W ignore $scriptdir/wmem/series2stack.py \
        $dm3dir $datadir \
        -r '*.dm3' -O '.tif' -d 'uint16' -M

}


function get_cmd_downsample_slices {
    # Get the command for in-plane downsampling of tifs.

    echo python -W ignore $scriptdir/wmem/downsample_slices.py \
        $datadir/$subdir $datadir/${subdir}_$dspf$ds \
        -r '*.tif' -f $ds -M

}


function adapt_fiji_register {
    # Copy and adapt the script for registering the slices with Fiji.

    sed "s?SOURCE_DIR?$datadir/tif?;\
        s?TARGET_DIR?$datadir/$regname?;\
        s?REFNAME?$regref?;\
        s?TRANSF_DIR?$datadir/$regname/trans?g" \
        $scriptdir/wmem/fiji_register.py \
        > $datadir/fiji_register.py

}


function get_cmd_fiji_register {
    # Get the command for registering the slices with Fiji.

    echo $imagej --headless $datadir/fiji_register.py  # string var
    # echo imagej --headless $datadir/fiji_register.py  # alias

}


function get_cmd_series2stack {
    # Get the command for creating a h5 dataset from a tif series.

    echo python -W ignore $scriptdir/wmem/series2stack.py \
        $datadir/$regname $datadir/$dataset$opf.h5/$ods \
        -r '*.tif' -O '.h5' -d 'uint16' -e $ze $ye $xe

}


function get_cmd_splitblocks {
    # Get the command for splitting a h5 dataset into blocks.
    # Specifying numpy slices.

    echo python -W ignore $scriptdir/wmem/stack2stack.py \
        $datadir/$dataset$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        -D $z $Z $y $Y $x $X $vol_slice

}


function get_cmd_splitblocks_mpi {
    # Get the command for splitting a h5 dataset into blocks.
    # Specifying numpy slices.

    echo python -W ignore $scriptdir/wmem/splitblocks.py \
        $datadir/$dataset$ipf.h5/$ids \
        $dataset -p $bs $bs $bs -q $zm $ym $xm $args

}


function get_cmd_splitblocks_datastem {
    # Get the command for splitting a h5 dataset into blocks.
    # Specifying the block identifier.

    echo python -W ignore $scriptdir/wmem/stack2stack.py \
        $datadir/$dataset$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        -p $datastem -b $xo $yo $zo

}


function get_cmd_apply_ilastik {
    # Get the command for applying the ilastik pixel classifier.

    # echo "export LAZYFLOW_THREADS=16;"
    # echo "export LAZYFLOW_TOTAL_RAM_MB=110000;"
    echo "$ilastik --headless \\"
    echo "--preconvert_stacks \\"
    echo "--project=$datadir/$pixprob_trainingset.ilp \\"
    echo "--output_axis_order=zyxc \\"
    echo "--output_format='compressed hdf5' \\"
    echo "--output_filename_format=$datadir/$dataset$opf.h5 \\"
    echo "--output_internal_path=$ods \\"
    echo "$datadir/$dataset$ipf.h5/$ids"

}


function write_ilastik_correct_attributes {
    # Write script to correct the element_size_um attribute.

    local pyfile=$1

    echo "import os" > $pyfile
    echo "from wmem import utils, Image, LabelImage" >> $pyfile
    echo "datadir = '$datadir'" >> $pyfile
    echo "dataset = '$dataset'" >> $pyfile
    echo "h5dset_in = dataset + '.h5/data'" >> $pyfile
    echo "h5path_in = os.path.join(datadir, h5dset_in)" >> $pyfile
    echo "im1 = utils.get_image(h5path_in)" >> $pyfile
    echo "h5dset_out = dataset + '_probs.h5/volume/predictions'" >> $pyfile
    echo "h5path_out = os.path.join(datadir, h5dset_out)" >> $pyfile
    echo "im2 = utils.get_image(h5path_out)" >> $pyfile
    echo "im2.elsize[:3] = im1.elsize[:3]" >> $pyfile
    echo "im2.h5_write_elsize()" >> $pyfile
    echo "im1.close()" >> $pyfile
    echo "im2.close()" >> $pyfile
    chmod u+x $pyfile

}


function get_cmd_sum_volumes {
    # Get the command for summing volumes of a zyxc h5 dataset.

    echo python -W ignore $scriptdir/wmem/combine_vols.py \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        -i $vols

}


function get_cmd_eed_deployed {
    # Get the command for filtering data with edge-enhancing diffusion.

    # echo $scriptdir/bin/run_EM_eed_simple.sh \  # does not run on ARCUSB
    echo $scriptdir/bin/EM_eed_simple \
        \'$datadir/blocks_${bs}\' \
        \'$datastem$ipf\' \'/$ids\' \
        \'$datastem$opf\' \'/$ods\' \
        \'0\' \'50\' \'1\' \'1\' \
        \> $datadir/blocks_$bs/${datastem}_$jobname.log

}


function get_cmd_eed_matlab {
    # Get the command for filtering data with edge-enhancing diffusion.

    local mlcmd
    mlcmd+=addpath\(\'$scriptdir/wmem\'\)\;
    mlcmd+=EM_eed_simple\(
    mlcmd+=\'$datadir/blocks_$bs\',
    mlcmd+=\'$datastem$ipf\',\'/$ids\',
    mlcmd+=\'$datastem$opf\',\'/$ods\',
    mlcmd+='0,50,1,1); exit;'

    echo "matlab -singleCompThread -nodisplay -nosplash -nojvm -r \""$mlcmd"\" > $datadir/blocks_$bs/$datastem${opf}_${ods////-}.log"

}


# function get_cmd_eed_matlab {
#     # Get the command for filtering data with edge-enhancing diffusion.
#
#     local mlcmd
#     mlcmd+=try\;
#     mlcmd+=addpath\(\'$scriptdir\'\)\;
#     mlcmd+=EM_eed_simple\(
#     mlcmd+=\'$datadir/blocks_$bs\',
#     mlcmd+=\'datastem$ipf\',\'/$ids\',
#     mlcmd+=\'datastem$opf\\',\'/$ods\',
#     mlcmd+='0,50,1,1\);'
#     mlcmd+='catch;end;quit;'
#
#     echo matlab -nodesktop -nojvm -r \'"$mlcmd"\'
#
# }


function get_cmd_mergeblocks {
    # Get the command for merging blocks of data.

    # unset infiles
    # declare -a infiles
    # get_infiles

    echo python -W ignore $scriptdir/wmem/mergeblocks.py \
        "${infiles[@]}" \
        $datadir/$dataset$opf.h5/$ods \
        -b $zo $yo $xo \
        --blocksize $zs $ys $xs \
        --blockmargin $zm $ym $xm \
        -s $zmax $ymax $xmax \
        $args

}


function get_cmd_prob2mask {
    # Get the command for thresholding the probability maps.
    # NOTE: arg='3 4 1 -g -l 0 -u 0 -s 2000 -d 1' for 4D slice

    echo python -W ignore $scriptdir/wmem/prob2mask.py \
        $datadir/$dataset$ipf.h5/$ids \
        $datadir/$dataset$opf.h5/$ods \
        -D $z $Z 1 0 0 1 0 0 1 $args

}


function get_cmd_prob2mask_datastems {
    # Get the command for thresholding the probability maps.

    echo python -W ignore $scriptdir/wmem/prob2mask.py \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        $arg

}


function get_cmd_downsample_blockwise {
    # Get the command for downsampling of h5 datasets.

    echo python -W ignore $scriptdir/wmem/downsample_blockwise.py \
        $datadir/$dataset$ipf.h5/$ids \
        $datadir/$dataset_ds$opf.h5/$ods \
        -B 1 $ds $ds $brvol -f $brfun \
        -D $z $Z 1 0 0 1 0 0 1 $vol_slice

}


function get_cmd_downsample_blockwise_expand {
    # Get the command for downsampling of h5 datasets.

    echo python -W ignore $scriptdir/wmem/downsample_blockwise.py \
        $datadir/$dataset_ds$ipf.h5/$ids \
        $datadir/$dataset$opf.h5/$ods \
        -B 1 $ds $ds $brvol -f $brfun \
        -D $z $Z 1 0 0 1 0 0 1 $vol_slice

}


function get_cmd_watershed {
    # Get the command for performing watershed on blocks.

    echo python -W ignore $scriptdir/wmem/watershed_ics.py \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        -l $l -u $u -s $s -S

}


function get_cmd_h52nii {
    # Get the command for converting h5 to nii.

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/stack2stack.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot${ipf}_$ods.nii.gz \
        $args

}


function get_cmd_conncomp_3D {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5/$ods \
        -m '3D' "$args"

}


function get_cmd_conncomp_2D {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5/$ods \
        -m '2D' -d 0 $mpiflag

}


function get_cmd_conncomp_2Dfilter {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5 \
        -m '2Dfilter' -d 0 $mpiflag \
        $MEANINT \
        -p ${props[@]}

}


function get_cmd_conncomp_2Dprops {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5 \
        -m '2Dprops' -d 0 $mpiflag \
        -b $datadir/$dataroot$opf \
        -p ${props[@]}

}


function get_cmd_conncomp_2Dto3D {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5/$ods \
        -m '2Dto3D' -d 0 $mpiflag

}


function get_cmd_conncomp_train {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components_clf.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5/$ods \
        -m 'train'  # TODO

}


function get_cmd_conncomp_test {
    # Get the command for .

    dataroot=$1

    echo python -W ignore $scriptdir/wmem/connected_components_clf.py \
        $clfpath $scalerpath \
        -m 'test' \
        -i $datadir/$dataroot$ipf.h5/$ids \
        -o $datadir/$dataroot$opf.h5/$ods \
        -b $datadir/$dataroot$opf \
        -p ${props[@]} $args

}


function get_cmd_slicvoxels {
    # Get the command for .

    echo python -W ignore $scriptdir/wmem/slicvoxels.py \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        -l $l -c $c -s $s -e

}


function get_cmd_s2s {
    # Get the command for converting h5.
    dataroot=$1
    echo python -W ignore $scriptdir/wmem/stack2stack.py \
        $datadir/$dataroot$ipf.h5/$ids \
        $datadir/$dataroot$opf.h5/$ods \
        $args

}


function get_cmd_smooth {
    # Get the command for .

    echo python -W ignore $scriptdir/wmem/image_ops.py \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        $args

}


function get_cmd_watershed_ics {
    # Get the command for .

    echo python -W ignore $scriptdir/wmem/watershed_ics.py \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        --masks NOT $datadir/blocks_$bs/$datastem$mpf.h5/$mds \
        $args

}


function get_cmd_agglo_mask {
    # Get the command for .

    echo python -W ignore $scriptdir/wmem/agglo_from_labelmask.py \
        $datadir/blocks_$bs/$datastem$lpf.h5/$lds \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        $args

}


function get_cmd_upsample_blocks {
    # Get the command for .

    coords=$( datastem2coords ${datastem} )
    coords_ds=$( downsampled_coords $coords )
    dataslices=$( coordsXYZ_to_slicesZYX $coords_ds )

    echo python -W ignore $scriptdir/wmem/downsample_blockwise.py \
        $datadir/${dataset_ds}${ipf}.h5/${ids} \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        -D "$dataslices" 1 -B 1 $ds $ds -f 'expand' -s $zmax $ymax $xmax

}


function get_cmd_fill_holes {
    # Get the command for .

    echo python -W ignore $scriptdir/wmem/fill_holes.py -S \
        $datadir/blocks_$bs/$datastem$ipf.h5/$ids \
        $datadir/blocks_$bs/$datastem$opf.h5/$ods \
        $args

}


function get_cmd_merge_labels_ws {
    # Get the command for .

    dataroot=$1

    echo python -W ignore "$scriptdir/wmem/merge_labels.py" \
        "$datadir/$dataroot$ipf.h5/$ids" \
        "$datadir/$dataroot$opf.h5/$ods" \
        "$args"

}


function get_cmd_NoR {
    # Get the command for .

    dataroot=$1

    echo python -W ignore "$scriptdir/wmem/nodes_of_ranvier.py" \
        "$datadir/$dataroot$ipf.h5/$ids" \
        "$datadir/$dataroot$opf.h5/$ods" \
        "$args"

}


function get_cmd_remap {
    # Get the command for .

    dataroot=$1

    echo python -W ignore "$scriptdir/wmem/remap_labels.py" \
        "$datadir/$dataroot$ipf.h5/$ids" \
        "$datadir/$dataroot$opf.h5/$ods" \
        "$args"

}


function get_cmd_combine_labels {
    # Get the command for .

    dataroot=$1

    echo python -W ignore "$scriptdir/wmem/combine_labels.py" \
        "$datadir/${dataroot}${ipf1}.h5/${ids1}" \
        "$datadir/${dataroot}${ipf2}.h5/${ids2}" \
        "$datadir/${dataroot}${opf}.h5/${ods}" \
        "$args"

}


function get_cmd_merge_slicelabels_mpi {
    # Get the command for .

    dataroot="$1"

    echo python -W ignore "$scriptdir/wmem/merge_slicelabels.py" \
        "$datadir/$dataroot$ipf.h5/$ids" \
        "$datadir/$dataroot$opf.h5/$ods" \
        "$args"

}


function get_cmd_merge_slicelabels {
    # Get the command for .

    dataroot="$1"

    echo python -W ignore "$scriptdir/wmem/merge_slicelabels.py" \
        "$datadir/$dataroot$ipf.h5/$ids" \
        "$datadir/$dataroot$opf.h5/$ods" \
        "$args"

}
