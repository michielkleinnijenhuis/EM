rsync -Pazv /Users/michielk/Downloads/ilastik-1.2.2post1-Linux.tar.bz2 ndcn0180@arcus-b.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace

rsync -Pazv /Users/michielk/oxdata/originaldata/P01/Myrf_00 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/originaldata/P01/EM
rsync -Pazv /Users/michielk/oxdata/originaldata/P01/Myrf_01 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/originaldata/P01/EM

rsync -Pazv /Volumes/MK_256GB/3Oct17/*.dm3 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/originaldata/P01/EM/Myrf_01/SET-B/3View/B-NT-S9-2a/

rsync -Pazv /Volumes/MK_256GB/26Oct17/B-NT-S10-2d jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/originaldata/P01/EM/Myrf_01/SET-B/3View/

rsync -Pazv /Volumes/MK_256GB/B-NT-S10-2f jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/originaldata/P01/EM/Myrf_01/SET-B/3View/

rsync -Pazv /vols/Data/km/michielk/originaldata/P01/EM/Myrf_00 ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM

rsync -Pazv /vols/Data/km/michielk/originaldata/P01/EM/Myrf_01 ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM

rsync -Pazv ~/workspace/pyDM3reader ndcn0180@arcus-b.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace


# FROM MK_256GB
scriptdir="$HOME/workspace/EM"
dm3dir="/Volumes/MK_256GB/B-NT-S10-2f"
dataset='B-NT-S10-2f'
datadir="$HOME/oxdata/P01/EM/Myrf_01_201708/SET-B/B-NT-S10-2f" && mkdir -p $datadir && cd $datadir
export PYTHONPATH=$PYTHONPATH:$scriptdir
export PYTHONPATH=$PYTHONPATH:/Users/michielk/workspace/pyDM3reader
basepath=$datadir/${dataset}
mpiexec -np 6 python $scriptdir/wmem/series2stack.py $dm3dir $datadir -o 'zyx' -s 5 20 20 -r '*.dm3' -O '.tif' -d 'uint16' -M
mpiexec -np 6 python $scriptdir/wmem/series2stack.py $dm3dir $datadir -o 'zyx' -s 5 20 20 -r '*.dm3' -O '.tif' -d 'uint16' -M -D 0 0 20 0 0 1 0 0 1


# LOCAL
scriptdir="$HOME/workspace/EM"
datadir="$HOME/oxdata/originaldata/P01/Myrf_00"
dataset='T4_1'
export PYTHONPATH=$PYTHONPATH:$scriptdir
export PYTHONPATH=$PYTHONPATH:/Users/michielk/workspace/pyDM3reader
basepath=$datadir/${dataset}
mpiexec -np 6 python $scriptdir/wmem/series2stack.py $datadir $basepath -o 'zyx' -s 5 20 20 -r '*.dm3' -O '.tif' -d 'uint16' -M
mpiexec -np 6 python $scriptdir/wmem/downsample_slices.py $basepath/jpg $basepath/jpg_ds -r '*.jpg' -f 7 -M

# ARC
export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
export PYTHONPATH=$scriptdir
export PYTHONPATH=$PYTHONPATH:$HOME/workspace/pyDM3reader
module load mpich2/1.5.3__gcc

# dataset parameters
dm3dir="${DATA}/EM/Myrf_00/3View/25Aug17/540slices_70nm_8k_25Aug17"
datadir="${DATA}/EM/Myrf_00" && cd $datadir
dataset='T4_1'
basepath=$datadir/${dataset}

datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
dataset='B-NT-S9-2a'
dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$dataset"
basepath=$datadir/${dataset}


datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
sample='B-NT-S10-2d'
dataset="${sample}"
dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$dataset"
basepath=$datadir/${dataset}

datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
sample='B-NT-S10-2d'
dataset="${sample}_setup"
# dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$dataset"
dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$sample/setup"
basepath=$datadir/${dataset}

datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
sample='B-NT-S10-2d'
dataset="${sample}_ROI_00"  # OK up to 130
dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$sample/ROI_00"
basepath=$datadir/${dataset}

datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
sample='B-NT-S10-2d'
dataset="${sample}_ROI_01"  # OK up to 15
dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$sample/ROI_01"
basepath=$datadir/${dataset}

datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
sample='B-NT-S10-2d'
dataset="${sample}_ROI_02"  # OK up to 240; is this continuous with ROI_00? NO
dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$sample/scan2_25Oct17"
basepath=$datadir/${dataset}

# datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
# dataset='B-NT-S10-2f_ROI_00'
# dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$dataset/ROI_00"
# basepath=$datadir/${dataset}
# datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
# dataset='B-NT-S10-2f_ROI_01'
# dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$dataset/ROI_01"
# basepath=$datadir/${dataset}
# datadir="${DATA}/EM/Myrf_01/SET-B" && cd $datadir
# dataset='B-NT-S10-2f_ROI_02'
# dm3dir="${DATA}/EM/Myrf_01/SET-B/3View/$dataset/ROI_02"
# basepath=$datadir/${dataset}


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
source datastems_blocks.sh


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2d_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 8649 Image Length: 8308


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2d_ROI_02'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 8844 Image Length: 8521

for tif in 'tif_ds'; do  # 'tif'
mkdir $datadir/$tif/artefacts
for i in `seq 4 9`; do
mv $datadir/$tif/002$i?.tif $datadir/$tif/artefacts
done
mv $datadir/$tif/003??.tif $datadir/$tif/artefacts
mv $datadir/$tif/004??.tif $datadir/$tif/artefacts
done


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 8423 Image Length: 8316


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_01'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 8649 Image Length: 8287


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_02'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 8457 Image Length: 8453


###=========================================================================###
### convert to tif
###=========================================================================###
module load mpich2/1.5.3__gcc

export template='single' additions='mpi'
export njobs=1 nodes=4 tasks=16 memcpu=2000 wtime="01:00:00" q=""
export jobname="dm3_convert"
scriptfile=$datadir/EM_dm3_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/series2stack.py \
$dm3dir $datadir -r '*.dm3' -O '.tif' -d 'uint16' -M" >> $scriptfile
export cmd="$scriptfile"
chmod +x $scriptfile
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### downsample
###=========================================================================###
module load mpich2/1.5.3__gcc

export template='single' additions='mpi'
export njobs=1 nodes=2 tasks=16 memcpu=10000 wtime="00:40:00" q=""
export jobname="ds"
scriptfile=$datadir/EM_ds_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/downsample_slices.py \
$datadir/tif $datadir/tif_ds -r '*.tif' -f 8 -M" >> $scriptfile
export cmd="$scriptfile"
chmod +x $scriptfile
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
### downsample
###=========================================================================###
module load mpich2/1.5.3__gcc

export template='single' additions='mpi'
export njobs=1 nodes=1 tasks=16 memcpu=10000 wtime="00:40:00" q=""
export jobname="ds"
scriptfile=$datadir/EM_ds_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/downsample_slices.py \
$datadir/reg $datadir/reg_ds -r '*.tif' -f 8 -M" >> $scriptfile
export cmd="$scriptfile"
chmod +x $scriptfile
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
$datadir/reg ${basepath}.h5/data -r '*.tif' -O '.h5' -d 'uint16' -e ${ze} ${ye} ${xe}" >> $scriptfile

export template='single' njobs=1 nodes=1 tasks=1 memcpu=30000 wtime='03:00:00' q='' jobname='h5' cmd="$scriptfile"
chmod +x $scriptfile
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### split in blocks
###=========================================================================###
# need parallel hdf5 reads here?
# module load hdf5-parallel/1.8.14_mvapich2_gcc
# without parallel: adapt script: FIXME: hold the job submission
mkdir -p $datadir/blocks
export template='array' additions='conda' CONDA_ENV='root'
export njobs=9 nodes=1 tasks=9 memcpu=6000 wtime='00:10:00' q='d'
export jobname='split'
export cmd="python $scriptdir/wmem/stack2stack.py \
${basepath}.h5/data $datadir/blocks/datastem.h5/data -p datastem"
source $scriptdir/pipelines/template_job_$template.sh

for i in `seq 0 8`; do
sed -i -e 's/node=9/node=1/g' EM_split_$i.sh
sed -i -e 's/ &//g' EM_split_$i.sh
sed -i -e 's/wait//g' EM_split_$i.sh
sbatch -p devel EM_split_$i.sh
done


###=========================================================================###
### train ilastik classifier
###=========================================================================###
mkdir -p $datadir/blocks
export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime="00:10:00" q="d"
export jobname="train"
datastems=( "${dataset}_03000-03500_03000-03500_00000-`printf %05d ${Z}`" )
export cmd="python $scriptdir/wmem/stack2stack.py \
${basepath}.h5/data $datadir/blocks/datastem.h5/data -p datastem"
source $scriptdir/pipelines/template_job_$template.sh

dataset='B-NT-S9-1a'; Z=479;
dataset='B-NT-S10-2d_ROI_00'; Z=135;
dataset='B-NT-S10-2d_ROI_02'; Z=240;
dataset='B-NT-S10-2f_ROI_00'; Z=184;
dataset='B-NT-S10-2f_ROI_01'; Z=184;
dataset='B-NT-S10-2f_ROI_02'; Z=184;
rem_host='ndcn0180@arcus-b.arc.ox.ac.uk'
rem_datadir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
loc_datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
fname="blocks/${dataset}_03000-03500_03000-03500_00000-`printf %05d ${Z}`.h5"
rsync -Pazv $rem_host:$rem_datadir/$fname $loc_datadir/blocks/

scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
alias ilastik=/Applications/ilastik-1.2.2post1-OSX.app/Contents/MacOS/ilastik
ilastik &

fname='pixclass.ilp'
fname='pixclass_8class.ilp'
rsync -Pazv $loc_datadir/$fname $rem_host:$rem_datadir

###=========================================================================###
### apply ilastik classifier
###=========================================================================###
pixprob_trainingset="pixclass_8class"
cp $basedir/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00_03000-03500_03000-03500_00000-00184.h5 $datadir/blocks/
cp $basedir/B-NT-S10-2f_ROI_00/$pixprob_traningset.ilp $datadir/

# array job on blocks
# nblocks=81
# datastems=("${datastems[@]:0:nblocks}")
source datastems_blocks.sh
datastems=( B-NT-S10-2f_ROI_00_00950-02050_04950-06050_00000-00184 B-NT-S10-2f_ROI_00_01950-03050_06950-08050_00000-00184 B-NT-S10-2f_ROI_00_02950-04050_05950-07050_00000-00184 )
datastems=( B-NT-S10-2f_ROI_00_03000-03500_03000-03500_00000-00184 )
nblocks=`echo "${datastems[@]}" | wc -w`

scriptfile=$datadir/EM_ilastik_script.sh
echo '#!/bin/bash' > $scriptfile
echo "#SBATCH --nodes=1" >> $scriptfile
echo "#SBATCH --ntasks-per-node=16" >> $scriptfile
echo "#SBATCH --mem-per-cpu=60000" >> $scriptfile
echo "#SBATCH --time=00:10:00" >> $scriptfile
echo "#SBATCH --job-name=EM_ilastik" >> $scriptfile
echo "export datastems=( "${datastems[@]}" )" >> $scriptfile
echo "export LAZYFLOW_THREADS=16; export LAZYFLOW_TOTAL_RAM_MB=60000;" >> $scriptfile
echo "$ilastik --headless \
--preconvert_stacks \
--project=$datadir/$pixprob_trainingset.ilp \
--output_axis_order=zyxc \
--output_format='compressed hdf5' \
--output_filename_format=$datadir/blocks/\${datastems[\$SLURM_ARRAY_TASK_ID]}_probs.h5 \
--output_internal_path=volume/predictions \
$datadir/blocks/\${datastems[\$SLURM_ARRAY_TASK_ID]}.h5/data" >> $scriptfile
chmod +x $scriptfile
sbatch --array=0-$((nblocks-1)) -p devel $scriptfile

# on the full stack
export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=16 memcpu=125000 wtime="23:00:00" q=""
export jobname="ilastikfull"
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

# TODO: correct element_size_um attribute

###=========================================================================###
### split _probs in blocks_500
###=========================================================================###
# need parallel hdf5 reads here?
# module load hdf5-parallel/1.8.14_mvapich2_gcc
# without parallel: adapt script: hold the job submission q='h'
mkdir -p $datadir/blocks_500
source datastems_blocks_500.sh
nblocks=`echo "${datastems[@]}" | wc -w`  # 289
export template='array' additions='conda' CONDA_ENV='root'
export njobs=33 nodes=1 tasks=9 memcpu=6000 wtime='00:10:00' q='h'
export jobname='split'
export cmd="python $scriptdir/wmem/stack2stack.py \
${basepath}_probs.h5/volume/predictions $datadir/blocks_500/datastem_probs.h5/volume/predictions -p datastem"
source $scriptdir/pipelines/template_job_$template.sh


for i in `seq 0 32`; do
sed -i -e 's/node=9/node=1/g' EM_split_$i.sh
sed -i -e 's/ &//g' EM_split_$i.sh
sed -i -e 's/wait//g' EM_split_$i.sh
# sbatch -p devel EM_split_$i.sh
done

sbatch -p devel EM_split_0.sh
for i in `seq 1 32`; do
sbatch EM_split_$i.sh
done

###=========================================================================###
### EED
###=========================================================================###
# rem_host='ndcn0180@arcus-b.arc.ox.ac.uk'
# rem_datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a'
# loc_datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S9-2a'
# fname='blocks/B-NT-S9-2a_00000-01050_00000-01050_00000-00479_probs.h5'
# rsync -Pazv $rem_host:$rem_datadir/$fname $loc_datadir/blocks

module load hdf5-parallel/1.8.17_mvapich2_gcc
module load matlab/R2015a

mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/wmem/EM_eed_simple.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
cd $datadir

# on blocks_500 (from blocks_probs)
# export template='array' additions=''
# export njobs=18 nodes=1 tasks=16 memcpu=50000 wtime='01:10:00' q=''
# export jobname='eed00'
# export cmd="$datadir/bin/EM_eed_simple \
# '$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
# '2' '50' '1' '1' > $datadir/blocks_500/datastem_probs_$jobname.log"
# source $scriptdir/pipelines/template_job_$template.sh

source datastems_blocks_500.sh
source find_missing_datastems.sh '_probs0_eed2' 'h5' ${datadir}/blocks_500/
nstems=${#datastems[@]}
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions=''
export nodes=1 memcpu=50000 wtime='01:10:00' q=''
export jobname='eed0'
export cmd="$datadir/bin/EM_eed_simple \
'$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
'1' '50' '1' '1' > $datadir/blocks_500/datastem_probs0_$jobname.log"
source $scriptdir/pipelines/template_job_$template.sh
source datastems_blocks_500.sh

source datastems_blocks_500.sh
source find_missing_datastems.sh '_probs1_eed2' 'h5' ${datadir}/blocks_500/
nstems=${#datastems[@]}
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions=''
export nodes=1 memcpu=50000 wtime='01:10:00' q=''
export jobname='eed1'
export cmd="$datadir/bin/EM_eed_simple \
'$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
'2' '50' '1' '1' > $datadir/blocks_500/datastem_probs_$jobname.log"
source $scriptdir/pipelines/template_job_$template.sh
source datastems_blocks_500.sh

source datastems_blocks_500.sh
source find_missing_datastems.sh '_probs2_eed2' 'h5' ${datadir}/blocks_500/
nstems=${#datastems[@]}
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions=''
export nodes=1 memcpu=50000 wtime='01:10:00' q=''
export jobname='eed2'
export cmd="$datadir/bin/EM_eed_simple \
'$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
'3' '50' '1' '1' > $datadir/blocks_500/datastem_probs_$jobname.log"
source $scriptdir/pipelines/template_job_$template.sh
source datastems_blocks_500.sh

# TODO: correct element_size_um and DIMENSION_LABELS attributes

###=========================================================================###
### merge blocks
###=========================================================================###
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime='03:10:00' q=''
export jobname="merge${pf}"

infiles=()
for file in `ls $datadir/blocks_500/*00184_probs2_eed2.h5`; do
    infiles+=("${file}/probs_eed")
done
export cmd="python $scriptdir/wmem/mergeblocks.py \
"${infiles[@]}" $datadir/${dataset}_probs2_eed2.h5/probs_eed \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### maskDS, maskMM, maskMM-0.02, maskMB
###=========================================================================###
module load mpich2/1.5.3__gcc
# module load hdf5-parallel/1.8.17_mvapich2_gcc

source datastems_blocks_500.sh
nstems=`echo "${datastems[@]}" | wc -w`  # 289
export template='array' additions='conda' CONDA_ENV='root'
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export nodes=1 memcpu=6000 wtime='00:10:00' q='d'
export jobname='maskDS'
export cmd="python $scriptdir/wmem/prob2mask.py ${datadir}/blocks_500/datastem.h5/data ${datadir}/blocks_500/datastem_masks.h5/maskDS -l 0 -u 10000000"
source $scriptdir/pipelines/template_job_$template.sh
export jobname='maskMM'
export cmd="python $scriptdir/wmem/prob2mask.py ${datadir}/blocks_500/datastem_probs0_eed2.h5/probs_eed ${datadir}/blocks_500/datastem_masks.h5/maskMM -l 0.2"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### merge blocks
###=========================================================================###
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime='00:10:00' q='d'

export jobname="mergeDS"
infiles=()
for file in `ls ${datadir}/blocks_500/*_masks.h5`; do
    infiles+=("${file}/maskDS")
done
export cmd="python $scriptdir/wmem/mergeblocks.py \
"${infiles[@]}" $datadir/${dataset}_masks.h5/maskDS \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax"
source $scriptdir/pipelines/template_job_$template.sh

export jobname="mergeMM"
infiles=()
for file in `ls ${datadir}/blocks_500/*_masks.h5`; do
    infiles+=("${file}/maskMM")
done
export cmd="python $scriptdir/wmem/mergeblocks.py \
"${infiles[@]}" $datadir/${dataset}_masks.h5/maskMM \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### blockreduce
###=========================================================================###

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"
export jobname="rb_${dspf}${pf}"
for pf in 'maskDS' 'maskMM'; do
    export jobname="rb_${dspf}${pf}"
    export cmd="python $scriptdir/wmem/downsample_blockwise.py \
    $datadir/${dataset}_masks.h5/$pf $datadir/${dataset}${dspf}${ds}_masks.h5/$pf \
    -B 1 ${ds} ${ds} -f 'np.amax'"
    source $scriptdir/pipelines/template_job_$template.sh
done

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="10:10:00" q=""
export jobname="rb_${dspf}${pf}"
export cmd="python $scriptdir/wmem/downsample_blockwise.py \
$datadir/${dataset}.h5/data $datadir/${dataset}${dspf}${ds}.h5/data \
-B 1 ${ds} ${ds} -f 'np.amax'"  # np.mean # memory error
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
$datadir/${dataset}${dspf}${ds}_masks.h5/maskMM $datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
-m '2D' -d 0"
source $scriptdir/pipelines/template_job_$template.sh
# FIXME: need to use maskDS here to get the borderaxons

### apply criteria to create forward mappings
export template='single' additions='conda' CONDA_ENV="root"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="10:10:00" q=""
export jobname="cc2Dfilter"
export cmd="python $scriptdir/wmem/connected_components.py \
$datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
$datadir/${dataset}${dspf}${ds}_labels.h5 \
-m '2Dfilter' -d 0 \
-p 'label' 'area' 'eccentricity' 'solidity' 'extent' 'euler_number' \
-a 10 -A 1500 -e 0 -s 0.50"
source $scriptdir/pipelines/template_job_$template.sh
# ### apply criteria to create forward mappings (MPI)
# module load hdf5-parallel/1.8.17_mvapich2_gcc  # PARALLEL H5 WITH MPI NOT WORKING
# export template='single' additions='mpi conda' CONDA_ENV="root"
# export njobs=1 nodes=2 tasks=16 memcpu=60000 wtime="00:10:00" q="d"
# export jobname="cc2Dfilter"
# export cmd="python $scriptdir/wmem/connected_components.py \
# $datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
# $datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_filter2D -M \
# -m '2Dfilter' -d 0 \
# -p 'label' 'area' 'eccentricity' 'solidity' 'extent' 'euler_number' \
# -a 10 -A 1500 -e 0 -E 0.30 -s 0.50"
# source $scriptdir/pipelines/template_job_$template.sh
### forward map
export template='single' additions='conda' CONDA_ENV="root"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"
export jobname="cc2Dprops"
export cmd="python $scriptdir/wmem/connected_components.py \
$datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
$datadir/${dataset}${dspf}${ds}_labels.h5 \
-b $datadir/${dataset}${dspf}${ds}_labels \
-m '2Dprops' -d 0 \
-p 'label' 'area' 'eccentricity' 'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh
### 3D labeling
export template='single' additions='conda' CONDA_ENV="root"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"
export jobname="cc2Dto3D"
export cmd="python $scriptdir/wmem/connected_components.py \
$datadir/${dataset}${dspf}${ds}_labels.h5/label \
$datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_2Dlabeled \
-m '2Dto3D' -d 0"
source $scriptdir/pipelines/template_job_$template.sh








# TypeError: Can't broadcast (184, 1188, 1204) -> (184, 8316, 8423)

import h5py
import numpy as np

def get_new_sizes(func, blockreduce, dssize, elsize):
    """Calculate the reduced dataset size and voxelsize."""
    if func == 'expand':
        fun_dssize = lambda d, b: int(np.ceil(float(d) * b))
        fun_elsize = lambda e, b: float(e) / b
    else:
        fun_dssize = lambda d, b: int(np.ceil(float(d) / b))
        fun_elsize = lambda e, b: float(e) * b
    dssize = [fun_dssize(d, b) for d, b in zip(dssize, blockreduce)]
    elsize = [fun_elsize(e, b) for e, b in zip(elsize, blockreduce)]
    return dssize, elsize

h5path_in = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00_masks.h5'
h5file = h5py.File(h5path_in, 'a')
ds_in = h5file['maskDS']
elsize = ds_in.attrs['element_size_um']
outsize, elsize = get_new_sizes('np.amax', [1, 7, 7], ds_in.shape, elsize)


# export template='array' additions='' CONDA_ENV=''
# export njobs=18 nodes=1 tasks=5 memcpu=60000 wtime="05:00:00" q="d"
# export jobname='maskDS'
# source datastems_blocks.sh
# rm -rf EM_$jobname*
# export xs=2000; export ys=2000;
# mkdir -p $datadir/EM_$jobname
# i=0
# for x in `seq 0 $xs $xmax`; do
#     [ $x == $(((xmax/xs)*xs)) ] && X=$xmax || X=$((x+xs))
#     for y in `seq 0 $ys $ymax`; do
#         [ $y == $(((ymax/ys)*ys)) ] && Y=$ymax || Y=$((y+ys))
#         echo "python $scriptdir/wmem/prob2mask.py ${datadir}/${dataset}.h5/data ${datadir}/${dataset}_maskDS.h5/maskDS -l 0 -u 10000000 -D $z $Z 1 $y $Y 1 $x $X 1" >> $datadir/EM_$jobname/EM_${jobname}_0.sh
#         i=$((i+1))
#     done
# done
#
#
# # module load mpich2/1.5.3__gcc
#
# njobs=`ls EM_$jobname/EM_${jobname}_*.sh | wc -l`
# export nodes=1 tasks=1 memcpu=60000 wtime='00:10:00'
# scriptfile=$datadir/EM_$jobname.sh
# echo '#!/bin/bash' > $scriptfile
# echo "#SBATCH --nodes=$nodes" >> $scriptfile
# echo "#SBATCH --ntasks-per-node=$tasks" >> $scriptfile
# echo "#SBATCH --mem-per-cpu=$memcpu" >> $scriptfile
# echo "#SBATCH --time=$wtime" >> $scriptfile
# echo "#SBATCH --job-name=$jobname" >> $scriptfile
# # echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
# # echo "source activate parallel" >> $scriptfile
# echo ". $datadir/EM_$jobname/EM_${jobname}_\${SLURM_ARRAY_TASK_ID}.sh" >> $scriptfile
# chmod +x $scriptfile
# sbatch -p devel --array=0-$((njobs-1)) $scriptfile


export jobname='maskDS'
rm -rf EM_$jobname*
export nodes=1 tasks=1 memcpu=60000 wtime='01:10:00'
scriptfile=$datadir/EM_$jobname.sh
echo '#!/bin/bash' > $scriptfile
echo "#SBATCH --nodes=$nodes" >> $scriptfile
echo "#SBATCH --ntasks-per-node=$tasks" >> $scriptfile
echo "#SBATCH --mem-per-cpu=$memcpu" >> $scriptfile
echo "#SBATCH --time=$wtime" >> $scriptfile
echo "#SBATCH --job-name=$jobname" >> $scriptfile
echo "python $scriptdir/wmem/prob2mask.py ${datadir}/${dataset}.h5/data ${datadir}/${dataset}_maskDS.h5/maskDS -l 0 -u 10000000" >> $scriptfile
chmod +x $scriptfile
sbatch $scriptfile

# export jobname='maskMM'
# rm -rf EM_$jobname*
# export nodes=1 tasks=1 memcpu=60000 wtime='01:10:00'
# scriptfile=$datadir/EM_$jobname.sh
# echo '#!/bin/bash' > $scriptfile
# echo "#SBATCH --nodes=$nodes" >> $scriptfile
# echo "#SBATCH --ntasks-per-node=$tasks" >> $scriptfile
# echo "#SBATCH --mem-per-cpu=$memcpu" >> $scriptfile
# echo "#SBATCH --time=$wtime" >> $scriptfile
# echo "#SBATCH --job-name=$jobname" >> $scriptfile
# echo "python $scriptdir/wmem/prob2mask.py ${datadir}/${dataset}_probs.h5/volume/predictions ${datadir}/${dataset}_maskMM_tmp.h5/maskMM -l 0.2 -D 0 0 1 0 0 1 0 0 1 0 1 1" >> $scriptfile
# chmod +x $scriptfile
# sbatch $scriptfile

export jobname='maskMM'
rm -rf EM_$jobname*
export nodes=1 tasks=1 memcpu=60000 wtime='01:10:00'
scriptfile=$datadir/EM_$jobname.sh
echo '#!/bin/bash' > $scriptfile
echo "#SBATCH --nodes=$nodes" >> $scriptfile
echo "#SBATCH --ntasks-per-node=$tasks" >> $scriptfile
echo "#SBATCH --mem-per-cpu=$memcpu" >> $scriptfile
echo "#SBATCH --time=$wtime" >> $scriptfile
echo "#SBATCH --job-name=$jobname" >> $scriptfile
echo "python $scriptdir/wmem/prob2mask.py ${datadir}/${dataset}_probs0_eed2.h5/probs_eed ${datadir}/${dataset}_maskMM.h5/maskMM -l 0.2" >> $scriptfile
chmod +x $scriptfile
sbatch $scriptfile







# export jobname="maskMM"
# export cmd="python $scriptdir/convert/prob2mask.py \
# $datadir datastem -m '_maskDS' 'stack' \
# -p '_probs0_eed2' 'stack' -l 0.2 -s 100000 -d 1 -o '_maskMM'"
# source $scriptdir/pipelines/template_job_$template.sh
#
# export jobname="maskMM002"
# export cmd="python $scriptdir/convert/prob2mask.py \
# $datadir datastem -m '_maskDS' 'stack' \
# -p '_probs0_eed2' 'stack' -l 0.02 -o '_maskMM-0.02'"
# source $scriptdir/pipelines/template_job_$template.sh
#
# export jobname="maskMB"
# export cmd="python $scriptdir/convert/prob2mask.py \
# $datadir datastem \
# -p '_probs' 'volume/predictions' -c 3 -l 0.3 -o '_maskMB'"
# source $scriptdir/pipelines/template_job_$template.sh
#
# for datastem in ${datastems[@]}; do
# echo $datastem
# python $scriptdir/convert/prob2mask.py \
# $datadir $datastem -m '_maskDS' 'stack' \
# -p '_probs0_eed2' 'stack' -l 0.2 -s 100000 -d 1 -o '_maskMM'
# done


###=========================================================================###



export jobname='st2st'
rm -rf EM_$jobname*
export nodes=1 tasks=1 memcpu=60000 wtime='02:10:00'
scriptfile=$datadir/EM_$jobname.sh
echo '#!/bin/bash' > $scriptfile
echo "#SBATCH --nodes=$nodes" >> $scriptfile
echo "#SBATCH --ntasks-per-node=$tasks" >> $scriptfile
echo "#SBATCH --mem-per-cpu=$memcpu" >> $scriptfile
echo "#SBATCH --time=$wtime" >> $scriptfile
echo "#SBATCH --job-name=$jobname" >> $scriptfile
echo "python $scriptdir/wmem/downsample_blockwise.py ${datadir}/${dataset}.h5/data ${datadir}/${dataset}_data_ds.h5/foo -B 8 8 1" >> $scriptfile
# echo "python $scriptdir/wmem/stack2stack.py ${datadir}/${dataset}.h5/data ${datadir}/${dataset}_data_ds.nii.gz -r 8 8 1 -o 'xyz'" >> $scriptfile
chmod +x $scriptfile
sbatch $scriptfile




export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"

module load mpi4py
module load hdf5-parallel/1.8.17_mvapich2_gcc

# module load hdf5-parallel/1.8.14_mvapich2_gcc
# module load python/2.7__gcc-4.8

# module load mpich2/1.5.3__gcc
# module load mpich2/1.5.3__gcc-4.4.7

export template='single' additions='mpi-conda' CONDA_ENV='h5para'
export njobs=1 nodes=2 tasks=6 memcpu=8000 wtime="00:10:00" q="d"
export jobname="tif_convert"
scriptfile=$datadir/EM_tif_convert_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/series2stack.py \
$datadir/$regname ${basepath}_testmpi.h5/data -r '*.tif' -O '.h5' -d 'uint16' -e ${ze} ${ye} ${xe} -M" >> $scriptfile
export cmd="$scriptfile"
chmod +x $scriptfile
source $scriptdir/pipelines/template_job_$template.sh


source activate scikit-image-devel_0.13
scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
DATA='/Users/michielk/oxdata/P01'
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S9-2a'
datadir=$basedir/${dataset}
cd $datadir
regname="reg00250"
refname="00250.tif"
xe=0.007; ye=0.007; ze=0.07
basepath=$datadir/${dataset}
mpiexec -np 2 python $scriptdir/wmem/series2stack.py $datadir/${regname}_ds ${basepath}_testmpi.h5/data -r '*.tif' -O '.h5' -d 'uint16' -e ${ze} ${ye} ${xe} -M



${HOME}/workspace/ilastik-1.2.2post1-Linux/run_ilastik.sh --debug


rem_host='ndcn0180@arcus-b.arc.ox.ac.uk'
rem_datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B'
loc_datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B'
fname='reg_ds'
for dset in 'B-NT-S9-2a' 'B-NT-S10-2d_ROI_00' 'B-NT-S10-2d_ROI_02' \
'B-NT-S10-2f_ROI_00' 'B-NT-S10-2f_ROI_01' 'B-NT-S10-2f_ROI_02'; do
rsync -Pazv $rem_host:$rem_datadir/$dset/$fname $loc_datadir/$dset/
done



# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_00//3View/25Aug17/540slices_70nm_8k_25Aug17/T4_1__3VBSED_slice_0000.dm3 /Users/michielk/oxdata/originaldata/P01/Myrf_00
# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_00/T4_1/tif_ds /Users/michielk/oxdata/P01/EM/Myrf_00/T4_1
# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a/tif_ds /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S9-2a

rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a/reg_ds /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S9-2a

# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2d_setup/tif /Users/michielk/oxdata/P01/EM/Myrf_01_201708/SET-B/B-NT-S10-2d_setup/
# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2d_ROI_00/tif_ds /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2d_ROI_00/
# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2d_ROI_01/tif_ds /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2d_ROI_01/
# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2d_ROI_02/tif_ds /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2d_ROI_02/

rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184* /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_500

rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_masks.h5  /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00
