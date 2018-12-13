###=========================================================================###
### prepare environment
###=========================================================================###
export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
export scriptdir="${HOME}/workspace/EM"
export PYTHONPATH=$scriptdir
export PYTHONPATH=$PYTHONPATH:$HOME/workspace/pyDM3reader
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64  # 20170515
ilastik=$HOME/workspace/ilastik-1.2.2post1-Linux/run_ilastik.sh


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S9-2a'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8649 Image Length: 8308
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2d_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8844 Image Length: 8521
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2d_ROI_02'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh

for tif in 'tif_ds'; do  # 'tif'
    mkdir $datadir/$tif/artefacts
    for i in `seq 4 9`; do
        mv $datadir/$tif/002$i?.tif $datadir/$tif/artefacts
    done
    mv $datadir/$tif/003??.tif $datadir/$tif/artefacts
    mv $datadir/$tif/004??.tif $datadir/$tif/artefacts
done


###=========================================================================###
### dataset parameters # Image Width: 8423 Image Length: 8316
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8649 Image Length: 8287
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_01'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8457 Image Length: 8453
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_02'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### convert to tif
###=========================================================================###
module load mpich2/1.5.3__gcc

scriptfile=$datadir/EM_dm3_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/series2stack.py \
$dm3dir $datadir -r '*.dm3' -O '.tif' -d 'uint16' -M" >> $scriptfile
chmod +x $scriptfile

export template='single' additions='mpi' CONDA_ENV=''
export njobs=1 nodes=4 tasks=16 memcpu=2000 wtime='01:00:00' q=''
export jobname='dm3_convert'
export cmd="$scriptfile"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### downsample (NOTE: should be done through downsample_blockwise.py now)
###=========================================================================###
module load mpich2/1.5.3__gcc

scriptfile=$datadir/EM_ds_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/downsample_slices.py \
$datadir/tif $datadir/tif_ds -r '*.tif' -f 8 -M" >> $scriptfile
chmod +x $scriptfile

export template='single' additions='mpi' CONDA_ENV=''
export njobs=1 nodes=2 tasks=16 memcpu=10000 wtime='00:40:00' q=''
export jobname='ds'
export cmd="$scriptfile"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### register
###=========================================================================###

mkdir -p $datadir/$regname/trans
sed "s?SOURCE_DIR?$datadir/tif?;\
    s?TARGET_DIR?$datadir/reg?;\
    s?REFNAME?$regref?;\
    s?TRANSF_DIR?$datadir/reg/trans?g" \
    $scriptdir/wmem/fiji_register.py \
    > $datadir/fiji_register.py

qsubfile=$datadir/fiji_register_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_reg" >> $qsubfile
echo "$imagej --headless \\" >> $qsubfile
echo "$datadir/fiji_register.py" >> $qsubfile
sbatch $qsubfile


###=========================================================================###
### downsample (NOTE: should be done through downsample_blockwise.py now)
###=========================================================================###
module load mpich2/1.5.3__gcc

scriptfile=$datadir/EM_ds_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/downsample_slices.py \
$datadir/reg $datadir/reg_ds -r '*.tif' -f 8 -M" >> $scriptfile
chmod +x $scriptfile

export template='single' additions='mpi' CONDA_ENV=''
export njobs=1 nodes=1 tasks=16 memcpu=10000 wtime='00:40:00' q=''
export jobname='ds'
export cmd="$scriptfile"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### convert to h5
###=========================================================================###
module load mpich2/1.5.3__gcc

scriptfile=$datadir/EM_h5_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "python $scriptdir/wmem/series2stack.py \
$datadir/reg ${basepath}.h5/data \
-r '*.tif' -O '.h5' -d 'uint16' -e ${ze} ${ye} ${xe}" >> $scriptfile
chmod +x $scriptfile

export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=1 memcpu=30000 wtime='03:00:00' q=''
export jobname='h5'
export cmd="$scriptfile"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### train ilastik classifier
###=========================================================================###

# get a small block of training data
mkdir -p $datadir/blocks
datastems=( "${dataset}_03000-03500_03000-03500_00000-`printf %05d ${Z}`" )
export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime='00:10:00' q='d'
export jobname='train'
export cmd="python $scriptdir/wmem/stack2stack.py \
    ${basepath}.h5/data $datadir/blocks/datastem.h5/data -p datastem"
source $scriptdir/pipelines/template_job_$template.sh

# sync the training data to LOCAL
dataset='B-NT-S9-1a'; Z=479;
dataset='B-NT-S10-2d_ROI_00'; Z=135;
dataset='B-NT-S10-2d_ROI_02'; Z=240;
dataset='B-NT-S10-2f_ROI_00'; Z=184;
dataset='B-NT-S10-2f_ROI_01'; Z=184;
dataset='B-NT-S10-2f_ROI_02'; Z=184;
rem_host='ndcn0180@arcus-b.arc.ox.ac.uk'
rem_datadir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
loc_datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
mkdir -p $loc_datadir/blocks
fname="blocks/${dataset}_03000-03500_03000-03500_00000-`printf %05d ${Z}`.h5"
rsync -Pazv $rem_host:$rem_datadir/$fname $loc_datadir/blocks/

# do interactive training on LOCAL
scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
alias ilastik=/Applications/ilastik-1.2.2post1-OSX.app/Contents/MacOS/ilastik
ilastik &

# send classifier to ARC
fname='pixclass_8class.ilp'
rsync -Pazv $loc_datadir/$fname $rem_host:$rem_datadir


###=========================================================================###
### apply ilastik classifier
###=========================================================================###
# copy the B-NT-S10-2f_ROI_00 trainingset for applying to current dataset
pixprob_trainingset="pixclass_8class"
trainingset='B-NT-S10-2f_ROI_00'
trainingblock='03000-03500_03000-03500_00000-00184'
cp $basedir/$trainingset/${trainingset}_$trainingblock.h5 $datadir/blocks/
cp $basedir/$trainingset/$pixprob_trainingset.ilp $datadir/

### apply classifier to the full stack
export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=16 memcpu=125000 wtime='23:00:00' q=''
export jobname='ilastik'
export cmd="export LAZYFLOW_THREADS=16; export LAZYFLOW_TOTAL_RAM_MB=110000;\
    $ilastik --headless \
    --preconvert_stacks \
    --project=$datadir/$pixprob_trainingset.ilp \
    --output_axis_order=zyxc \
    --output_format='compressed hdf5' \
    --output_filename_format=$datadir/${dataset}_probs.h5 \
    --output_internal_path=volume/predictions \
    $datadir/$dataset.h5/data"
source $scriptdir/pipelines/template_job_$template.sh

# correct element_size_um attribute  # TODO: make into function/module
pyfile=$datadir/EM_corr_script.py
echo "import os" > $pyfile
echo "import numpy as np" >> $pyfile
echo "from wmem import utils" >> $pyfile
echo "datadir = '$datadir'" >> $pyfile
echo "dataset = '$dataset'" >> $pyfile
echo "h5dset_in = dataset + '.h5/data'" >> $pyfile
echo "h5path_in = os.path.join(datadir, h5dset_in)" >> $pyfile
echo "h5_in, ds_in, es, al = utils.h5_load(h5path_in)" >> $pyfile
echo "h5dset_out = dataset + '_probs.h5/volume/predictions'" >> $pyfile
echo "h5path_out = os.path.join(datadir, h5dset_out)" >> $pyfile
echo "h5_out, ds_out, _, _ = utils.h5_load(h5path_out)" >> $pyfile
echo "es = np.append(ds_in.attrs['element_size_um'], 1)" >> $pyfile
echo "utils.h5_write_attributes(ds_out, element_size_um=es)" >> $pyfile
echo "h5_in.close()" >> $pyfile
echo "h5_out.close()" >> $pyfile

scriptfile=$datadir/EM_corr_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "python $pyfile" >> $scriptfile
chmod +x $scriptfile

export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=1 memcpu=30000 wtime='00:10:00' q='d'
export jobname='corr'
export cmd="$scriptfile"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### split _probs in blocks_0500
###=========================================================================###
module load hdf5-parallel/1.8.17_mvapich2_gcc

export template='array' additions='conda' CONDA_ENV='root'
export bs='0500' && source datastems_blocks_${bs}.sh
# source find_missing_datastems.sh '_probs' 'h5' ${datadir}/blocks_${bs}/
export nstems=${#datastems[@]}
export tasks=4
export njobs=$(( ($nstems + tasks-1) / $tasks))
export nodes=1 memcpu=6000 wtime='00:10:00' q='h'
export jobname='split'
export cmd="python $scriptdir/wmem/stack2stack.py \
    $datadir/${dataset}_probs.h5/volume/predictions \
    $datadir/blocks_${bs}/datastem_probs.h5/volume/predictions \
    -p datastem"
source $scriptdir/pipelines/template_job_$template.sh

for i in `seq 0 $((njobs-1))`; do
    sed -i -e "s/node=$tasks/node=1/g" EM_${jobname}_$i.sh
    sed -i -e "s/ &//g" EM_${jobname}_$i.sh
    sed -i -e "s/wait//g" EM_${jobname}_$i.sh
    # sbatch EM_${jobname}_$i.sh
    sbatch -p devel EM_${jobname}_$i.sh
done


###=========================================================================###
### combine vols
###=========================================================================###
module load hdf5-parallel/1.8.17_mvapich2_gcc

# myelin compartment [0:myelin_proper, 2:mito, 4:myelin_inner, 7:myelin_outer]
export template='array' additions='conda' CONDA_ENV='root'
export bs='0500' && source datastems_blocks_${bs}.sh
export nstems=${#datastems[@]}
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export nodes=1 memcpu=6000 wtime='00:10:00' q='d'
export jobname='sumvols0247'
export cmd="python $scriptdir/wmem/combine_vols.py \
    $datadir/blocks_${bs}/datastem_probs.h5/volume/predictions \
    $datadir/blocks_${bs}/datastem_probs.h5/sum0247 \
    -i 0 2 4 7"
source $scriptdir/pipelines/template_job_$template.sh

# ICS compartment [1:myelinated_axons, 6:unmyelinated_axons]
export template='array' additions='conda' CONDA_ENV='root'
export bs='0500' && source datastems_blocks_${bs}.sh
export nstems=${#datastems[@]}
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export nodes=1 memcpu=6000 wtime='00:10:00' q=''
export jobname='sumvols16'
export cmd="python $scriptdir/wmem/combine_vols.py \
    $datadir/blocks_${bs}/datastem_probs.h5/volume/predictions \
    $datadir/blocks_${bs}/datastem_probs.h5/sum16 \
    -i 1 6"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### edge enhancing diffusion filter
###=========================================================================###
module load hdf5-parallel/1.8.17_mvapich2_gcc
module load matlab/R2015a

mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh \
-m $scriptdir/wmem/EM_eed_simple.m \
-a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
cd $datadir

# for each of these:
h5_in='probs'; dset_in='volume/predictions'; h5_out='probs_eed'; dset_out='probs_eed'; wtime='10:10:00';
h5_in='probs'; dset_in='sum0247'; h5_out='probs_eed'; dset_out='sum0247_eed'; wtime='01:10:00';
h5_in='probs'; dset_in='sum16'; h5_out='probs_eed'; dset_out='sum16_eed'; wtime='01:10:00';

export bs='0500' && source datastems_blocks_${bs}.sh
source find_missing_h5.sh "${datadir}/blocks_${bs}/" "_$h5_out" "/$dset_out"
export nstems=${#datastems[@]} && echo $nstems
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions='' CONDA_ENV=''
export nodes=1 memcpu=50000 q=''
export jobname="$h5_out.$dset_out"
export cmd="$datadir/bin/EM_eed_simple \
    '$datadir/blocks_${bs}' \
    'datastem_${h5_in}' '/$dset_in' 'datastem_${h5_out}' '/$dset_out' \
    '0' '50' '1' '1' \
    > $datadir/blocks_${bs}/datastem_$jobname.log"
source $scriptdir/pipelines/template_job_$template.sh
source datastems_blocks_${bs}.sh


###=========================================================================###
### merge blocks
###=========================================================================###

function get_cmd_mergeblocks {
    get_infiles
    echo python $scriptdir/wmem/mergeblocks.py \
        "${infiles[@]}" $datadir/$dataset$ipf.h5/$ids \
        -b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax
}

export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime='01:10:00' q='h'
for ipf in '_probs'; do
    for ids in 'probs_eed' 'sum0247_eed' 'sum16_eed'; do
        export jobname="mEED$ipf.$ids"
        cmd=$(get_cmd_mergeblocks)
        source $scriptdir/pipelines/template_job_$template.sh
    done
done

###=========================================================================###
### maskDS, maskMM, maskMA, maskICS  # on full vol (without parallel, but not memory-intensive)
###=========================================================================###

function get_cmd_prob2mask {
    for z in `seq 0 $blocksize $zmax`; do
        get_Z
        echo python $scriptdir/wmem/prob2mask.py \
            $datadir/$dataset$ipf.h5/$ids \
            $datadir/$dataset$opf.h5/$ods \
            -g $arg -D $z $Z 1 0 0 1 0 0 1 $vol_slice
    done
}

# for blocksize: h5dump -pH $datadir/${dataset}${ipf}.h5 | grep CHUNKED

declare -a VARSETS
VARSETS[0]="ipf=; ids='data'; \
    opf='_masks'; ods='maskDS'; \
    arg='-l 0 -u 10000000'; blocksize=20; vol_slice=;"  # TODO: dilate?
VARSETS[1]="ipf='_probs_eed'; ids='sum0247_eed'; \
    opf='_masks'; ods='maskMM'; \
    arg='-l 0.5 -s 2000 -d 1 -S'; blocksize=20; vol_slice=;"
VARSETS[2]="ipf='_probs_eed'; ids='sum16_eed'; \
    opf='_masks'; ods='maskICS'; \
    arg='-l 0.2'; blocksize=20; vol_slice=;"
VARSETS[3]="ipf='_probs_eed'; ids='probs_eed'; \
    opf='_masks'; ods='maskMA'; \
    arg='-l 0.2'; blocksize=10; vol_slice='1 2 1';"  # NOTE: keep blocksize < 10

# submit a job
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime='00:10:00' q='d'

for vars in "${VARSETS[@]}"; do
    # set the variables
    awkfunction='{ for(i=0;i<=NF;i++) {printf "%s\n", $i} }'
    eval $( echo "${vars}" | awk -F ';' "$awkfunction" )
    # submit the job
    export jobname=$ods
    unset cmd && read -r -d '' cmd <<- EOM
    $(get_cmd_prob2mask)
EOM
    source $scriptdir/pipelines/template_job_$template.sh
done


# supply jobnamegenerator and function
function submit_jobs {
    for vars in "${VARSETS[@]}"; do
        # set the variables
        awkfunction='{ for(i=0;i<=NF;i++) {printf "%s\n", $i} }'
        eval $( echo "${vars}" | awk -F ';' "$awkfunction" )
        # submit the job
        export jobname="$2"
        unset cmd && read -r -d '' cmd <<- EOM
        $($1)
EOM
        source $scriptdir/pipelines/template_job_$template.sh
done
}
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime='00:10:00' q='d'
submit_jobs get_cmd_prob2mask '$ods'


###=========================================================================###
### blockreduce
###=========================================================================###

function get_cmd_downsample_blockwise {
    for z in `seq 0 $blocksize $zmax`; do
        get_Z
        echo python $scriptdir/wmem/downsample_blockwise.py \
            $datadir/$dataset$ipf.h5/$ids \
            $datadir/$dataset_ds$opf.h5/$ods \
            -B 1 $ds $ds $vol_br -f $fun \
            -D $z $Z 1 0 0 1 0 0 1 $vol_slice
    done
}

declare -a VARSETS
# downsample data with 'np.mean': work with datablocks
VARSETS[0]="ipf=; ids=data; \
    opf=; ods=data; \
    fun='np.mean'; vol_br=; \
    blocksize=20; vol_slice=; \
    memcpu=60000; wtime=02:00:00; q=;"
VARSETS[1]="ipf=_probs_eed; ids=sum0247_eed; \
    opf=_probs_eed; ods=sum0247_eed; \
    fun='np.mean'; vol_br=; \
    blocksize=20; vol_slice=; \
    memcpu=60000; wtime=02:00:00; q=;"
VARSETS[2]="ipf=_probs_eed; ids=sum16_eed; \
    opf=_probs_eed; ods=sum16_eed; \
    fun='np.mean'; vol_br=; \
    blocksize=20; vol_slice=; \
    memcpu=60000; wtime=02:00:00; q=;"
VARSETS[3]="ipf=_probs_eed; ids=probs_eed; \
    opf=_probs_eed; ods=probs_eed; \
    fun='np.mean'; vol_br=1; \
    blocksize=10; vol_slice='0 0 1'; \
    memcpu=125000; wtime=10:00:00; q=;"
VARSETS[4]="ipf=_probs; ids=volume/predictions; \
    opf=_probs; ods=volume/predictions; \
    fun='np.mean'; vol_br=1; \
    blocksize=10; vol_slice='0 0 1'; \
    memcpu=125000; wtime=10:00:00; q=;"
# # downsample masks with 'np.amax'; can work with the full volume
VARSETS[5]="ipf=_masks; ids=maskDS; \
    opf=_masks; ods=maskDS; \
    fun='np.amax'; vol_br=; \
    blocksize=$zmax; vol_slice=; \
    memcpu=60000; wtime=00:10:00; q='d';"
VARSETS[6]="ipf=_masks; ids=maskMM;
    opf=_masks; ods=maskMM; \
    fun='np.amax'; vol_br=; \
    blocksize=$zmax; vol_slice=; \
    memcpu=60000; wtime=00:10:00; q='d';"
VARSETS[7]="ipf=_masks; ids=maskMA; \
    opf=_masks; ods=maskMA; \
    fun='np.amax'; vol_br=; \
    blocksize=$zmax; vol_slice=; \
    memcpu=60000; wtime=00:10:00; q='d';"
VARSETS[8]="ipf=_masks; ids=maskICS; \
    opf=_masks; ods=maskICS; \
    fun='np.amax'; vol_br=; \
    blocksize=$zmax; vol_slice=; \
    memcpu=60000; wtime=00:10:00; q='d';"

export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1
for vars in "${VARSETS[@]}"; do
    # set the variables
    awkfunction='{ for(i=0;i<=NF;i++) {printf "%s\n", $i} }'
    eval $( echo "${vars}" | awk -F ';' "$awkfunction" )
    # submit the job
    export jobname="rb_$dspf$ds$ipf$ids"
    unset cmd && read -r -d '' cmd <<- EOM
    $(get_cmd_downsample_blockwise)
EOM
    source $scriptdir/pipelines/template_job_$template.sh
done


# ###=========================================================================###
# ### split in blocks
# ###=========================================================================###
# # need parallel hdf5 reads here?
# # module load hdf5-parallel/1.8.14_mvapich2_gcc
# # without parallel: hold job submission and adapt script

export bs=0500 nodes=1 tasks=9
# export bs=1000 nodes=1 tasks=9
# export bs=2000 nodes=1 tasks=1
pf=; dset='data';
# export pf='_probs1_eed2'; dset='probs_eed';
# pf='_masks'; dset='maskDS';
# pf='_masks'; dset='maskMM';

mkdir -p $datadir/blocks_${bs}
source datastems_blocks_${bs}.sh
source find_missing_datastems.sh '' 'h5' ${datadir}/blocks_${bs}/
nstems=${#datastems[@]}
echo $nstems
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions='conda' CONDA_ENV='root'
export nodes=1 memcpu=6000 wtime='00:10:00' q='h'
export jobname='split'
export cmd="python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset}${pf}.h5/${dset} \
$datadir/blocks_${bs}/datastem${pf}.h5/${dset} \
-p datastem"
source $scriptdir/pipelines/template_job_$template.sh

for i in `seq 0 $((njobs-1))`; do
sed -i -e "s/node=$tasks/node=1/g" EM_split_$i.sh
sed -i -e "s/ &//g" EM_split_$i.sh
sed -i -e "s/wait//g" EM_split_$i.sh
sbatch -p devel EM_split_$i.sh
done


###=========================================================================###
### watershed on prob_ics
###=========================================================================###
scriptdir="${HOME}/workspace/EM"

export template='array' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"

l=0.99; u=1.00; s=010;
svoxpf="_ws_l${l}_u${u}_${s}"
export jobname="ws"
export cmd="python $scriptdir/wmem/watershed_ics.py \
$datadir/blocks_2000/datastem_probs1_eed2.h5/probs_eed \
$datadir/blocks_2000/datastem_ws.h5/l0.99-u1.00-s010 \
-l $l -u $u -s $s -S"
source $scriptdir/pipelines/template_job_$template.sh

--masks NOT $datadir/${dataset}_masks.h5/maskMM \
XOR $datadir/${dataset}_masks.h5/maskDS \

###=========================================================================###
### watershed on prob_ics (MEM ERROR)
###=========================================================================###
scriptdir="${HOME}/workspace/EM"

export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"

l=0.99; u=1.00; s=010;
svoxpf="_ws_l${l}_u${u}_${s}"
export jobname="ws"
export cmd="python $scriptdir/wmem/watershed_ics.py \
$datadir/${dataset}_probs1_eed2.h5/probs_eed \
$datadir/${dataset}_ws.h5/l0.99-u1.00-s010 \
--masks NOT $datadir/${dataset}_masks.h5/maskMM \
XOR $datadir/${dataset}_masks.h5/maskDS \
-l $l -u $u -s $s"
source $scriptdir/pipelines/template_job_$template.sh












###=========================================================================###
### 2D connected components in maskMM
###=========================================================================###
# module load mpich2/1.5.3__gcc

dspf='ds'; ds=7;

### basic 2D labeling
export template='single' additions='conda' CONDA_ENV="root"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"
export jobname="cc2D"
export cmd="python $scriptdir/wmem/connected_components.py \
$datadir/${dataset}${dspf}${ds}_masks.h5/maskMM \
$datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
--maskDS $datadir/${dataset}${dspf}${ds}_masks.h5/maskDS \
-m '2D' -d 0"
source $scriptdir/pipelines/template_job_$template.sh
# FIXME: need to use maskDS here to get the borderaxons
# FIXME: EED seems to have shown serious boundary effects





###=========================================================================###
### Neuroproof ###
###=========================================================================###
CONDA_PATH="$(conda info --root)"
PREFIX="${CONDA_PATH}/envs/neuroproof-test"
NPdir="${HOME}/workspace/Neuroproof_minimal"
