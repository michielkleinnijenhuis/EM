# ssh -Y ndcn0180@arcus.arc.ox.ac.uk
# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
# remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
# localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test"
# localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
# rsync -avz $remdir/${f} $localdir
# rsync -avz $localdir/${f} $remdir

# source activate scikit-image-devel_0.13
# conda install h5py scipy
# pip install nibabel
###=========================================================================###
### environment prep
###=========================================================================###

export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/M3/M3_S1_GNU_old" && cd $datadir
source datastems_90blocks.sh


###=========================================================================###
### stitch (not performed again)
###=========================================================================###
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64

cd $datadir
mkdir -p $datadir/stitched

i=0
for z in `seq 0 46 414`; do
Z=$((z+46))
sed "s?INPUTDIR?$datadir/tifs?;\
    s?OUTPUTDIR?$datadir/stitched?;\
    s?Z_START?$z?;\
    s?Z_END?$Z/tifs?g" \
    $scriptdir/reg/EM_montage2stitched.py \
    > $datadir/EM_montage2stitched_`printf %03d $i`.py
    i=$((i+1))
done


###=========================================================================###
### register (performed again on previously stitched - ref to 0250.tif)
###=========================================================================###
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64
regname="reg0250"
refname="0250.tif"

cd $datadir
mkdir -p $datadir/$regname/trans

sed "s?SOURCE_DIR?$datadir/stitched?;\
    s?TARGET_DIR?$datadir/$regname?;\
    s?REFNAME?$refname?;\
    s?TRANSF_DIR?$datadir/$regname/trans?g" \
    $scriptdir/reg/EM_register.py \
    > $datadir/EM_register.py

qsubfile=$datadir/EM_register_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_reg" >> $qsubfile
echo "$imagej --headless \\" >> $qsubfile
echo "$datadir/EM_register.py" >> $qsubfile
sbatch $qsubfile


###=========================================================================###
### downsample
###=========================================================================###
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

dspf='ds'; ds=7;
export template='single' additions='mpi'
export njobs=1 nodes=1 tasks=16 memcpu=60000 wtime="00:10:00" q="d"
export jobname="ds"
export cmd="python $scriptdir/convert/EM_downsample.py \
$datadir/reg0250 $datadir/reg0250_${dspf}${ds} -d ${ds} -m"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### Create the stack (fails with mpi4py, but there's no need)
###=========================================================================###
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

export template='single' additions='mpi'
export njobs=1 nodes=1 tasks=1 memcpu=125000 wtime="01:00:00" q=""
export jobname="s2s"
export cmd="python $scriptdir/convert/EM_series2stack.py \
$datadir/reg0250 $datadir/${dataset}.h5 \
-f 'stack' -e ${xe} ${ye} ${ze} -c 20 20 20 -o"
source $scriptdir/pipelines/template_job_$template.sh

dspf='ds'; ds=7;
export template='single' additions='mpi'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:00:00" q="d"
export jobname="s2s_ds"
export cmd="python $scriptdir/convert/EM_series2stack.py \
$datadir/reg0250_${dspf}${ds} $datadir/${dataset}_${dspf}${ds}.h5 \
-f 'stack' -e $(echo $xe*$ds | bc) $(echo $ye*$ds | bc) $ze -c 20 20 20 -o"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### split volume in blocks
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=10 nodes=1 tasks=9 memcpu=6000 wtime="00:10:00" q="d"

export jobname="split"
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}.h5 \
$datadir/datastem.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx -p datastem"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### apply Ilastik classifier #
###=========================================================================###
# no need to load modules
pixprob_trainingset="pixprob_training"

export template='single' additions='conda' CONDA_ENV="ilastik-devel"
export njobs=90 nodes=1 tasks=16 memcpu=8000 wtime="01:30:00" q=""

export jobname="ilastik"
export cmd="${CONDA_PATH}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=zyxc \
--output_format=hdf5 \
--output_filename_format=$datadir/datastem_probs.h5 \
--output_internal_path=volume/predictions \
$datadir/datastem.h5/stack"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### EED
###=========================================================================###
module load hdf5-parallel/1.8.14_mvapich2_intel
module load python/2.7__gcc-4.8
module load matlab/R2015a

mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
cd $datadir

export template='array' additions=""
export njobs=1 nodes=1 tasks=1 memcpu=50000 wtime="05:00:00" q=""

layer=1
export jobname="EED${layer}_last"
export cmd="$datadir/bin/EM_eed \
'$datadir' 'datastem_probs' '/volume/predictions' '/stack' $layer \
> $datadir/datastem_probs0_eed2.log"
source $scriptdir/pipelines/template_job_$template.sh
layer=2
export jobname="EED${layer}"
export cmd="$datadir/bin/EM_eed \
'$datadir' 'datastem_probs' '/volume/predictions' '/stack' $layer \
> $datadir/datastem_probs1_eed2.log"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### to zyx(c) - moved originals to probs_backup
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=6000 wtime="00:30:00" q="d"
export jobname="zyx_p0"
pf='_probs0_eed2'
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 $datadir/datastem${pf}_zyx.h5 \
-e $ze $ye $xe -i 'xyz' -l 'zyx' \
-f 'stack'  -g 'stack'"
source $scriptdir/pipelines/template_job_$template.sh
# h5ls -v M3S1GNU_08950-09197_07950-08786_00030-00460_probs0_eed2.h5
# h5ls -v M3S1GNU_08950-09197_07950-08786_00030-00460_probs0_eed2_zyx.h5

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=6000 wtime="00:30:00" q="d"
export jobname="zyx_p1"
pf='_probs1_eed2'
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 $datadir/datastem${pf}_zyx.h5 \
-e $ze $ye $xe -i 'xyz' -l 'zyx' \
-f 'stack'  -g 'stack'"
source $scriptdir/pipelines/template_job_$template.sh
# h5ls -v M3S1GNU_08950-09197_07950-08786_00030-00460_probs0_eed2.h5
# h5ls -v M3S1GNU_08950-09197_07950-08786_00030-00460_probs0_eed2_zyx.h5

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=18 nodes=1 tasks=5 memcpu=125000 wtime="03:00:00" q=""
export jobname="zyxc_p"
pf='_probs'
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 $datadir/datastem${pf}_zyxc.h5 \
-e $ze $ye $xe 1 -i 'xyzc' -l 'zyxc' \
-f 'volume/predictions'  -g 'volume/predictions'"
source $scriptdir/pipelines/template_job_$template.sh
# h5ls -v M3S1GNU_08950-09197_07950-08786_00030-00460_probs.h5/volume
# h5ls -v M3S1GNU_08950-09197_07950-08786_00030-00460_probs_zyxc.h5/volume

###=========================================================================###
### base data block nifti's
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

export jobname="nifti"
pf=
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz \
-e $xe $ye $ze -i 'zyx' -l 'xyz' -u"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### maskDS, maskMM, maskMM-0.02, maskMB
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=18 nodes=1 tasks=5 memcpu=60000 wtime="05:00:00" q="d"
export jobname="maskDS"
export cmd="python $scriptdir/convert/prob2mask.py \
$datadir datastem \
-p \"\" 'stack' -l 0 -u 10000000 -o '_maskDS'"
source $scriptdir/pipelines/template_job_$template.sh

export jobname="maskMM"
export cmd="python $scriptdir/convert/prob2mask.py \
$datadir datastem -m '_maskDS' 'stack' \
-p '_probs0_eed2' 'stack' -l 0.2 -s 100000 -d 1 -o '_maskMM'"
source $scriptdir/pipelines/template_job_$template.sh

export jobname="maskMM002"
export cmd="python $scriptdir/convert/prob2mask.py \
$datadir datastem -m '_maskDS' 'stack' \
-p '_probs0_eed2' 'stack' -l 0.02 -o '_maskMM-0.02'"
source $scriptdir/pipelines/template_job_$template.sh

export jobname="maskMB"
export cmd="python $scriptdir/convert/prob2mask.py \
$datadir datastem \
-p '_probs' 'volume/predictions' -c 3 -l 0.3 -o '_maskMB'"
source $scriptdir/pipelines/template_job_$template.sh

for datastem in ${datastems[@]}; do
echo $datastem
python $scriptdir/convert/prob2mask.py \
$datadir $datastem -m '_maskDS' 'stack' \
-p '_probs0_eed2' 'stack' -l 0.2 -s 100000 -d 1 -o '_maskMM'
done

###=========================================================================###
### merge blocks
###=========================================================================###
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime="00:30:00" q=""

for pf in '_maskDS' '_maskMM' '_maskMM-0.02' '_maskMB'; do
    export jobname="merge${pf}"
    export cmd="python $scriptdir/convert/EM_mergeblocks.py \
    $datadir $datadir/${dataset}${pf}.h5 \
    -i ${datastems[*]} -t ${pf} \
    -f 'stack' -l 'zyx' \
    -b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax"
    source $scriptdir/pipelines/template_job_$template.sh
done
# full size _ws*
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=64GB wtime="01:00:00" q=""
pf='_ws_l0.99_u1.00_s005'
export jobname="merge${pf}"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax -l -r"
source $scriptdir/pipelines/template_job_$template.sh

### merge ws blocks (on arcus, NOT arcus-b); with block_reduce
# note that with blockreduce the file is also written to the same location as without!
# it should be about 18min per block
module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2
dspf='ds'; ds=7;
export template='single-pbs' additions='mpi'
export njobs=1 nodes=6 tasks=5 memcpu=64GB wtime="02:10:00" q=""
# export njobs=1 nodes=2 tasks=6 memcpu=50GB wtime="00:10:00" q="d"
pf='_ws_l0.99_u1.00_s005'
export jobname="merge${pf}_mpi"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' -o '${pf}_mpi' 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -F -B 1 ${ds} ${ds} -f 'mode' -m"
source $scriptdir/pipelines/template_job_$template.sh
# mv ${dataset}${pf}.h5 ${dataset}${dspf}${ds}${pf}.h5

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=10GB wtime="10:00:00" q=""
# export njobs=1 nodes=2 tasks=6 memcpu=50GB wtime="00:10:00" q="d"
pf='_ws_l0.99_u1.00_s010'
export jobname="merge${pf}_mode"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' -o "${pf}_mode2" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -F -B 1 ${ds} ${ds} -f 'mode'"
source $scriptdir/pipelines/template_job_$template.sh

dspf='ds'; ds=7;
# pf='_ws_l0.99_u1.00_005'
# for datastem in ${datastems[@]}; do
#     cp $datastem$pf.h5 $datastem${pf}_test.h5
# done
pf='_ws_l0.99_u1.00_s005'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack'  -o "${pf}_modetest" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -F -B 1 ${ds} ${ds} -f 'mode'

###=========================================================================###
### blockreduce
###=========================================================================###

dspf='ds'; ds=7;
export jobname="nifti${pf}"
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNU_${dspf}${ds}.h5 $datadir/M3S1GNU${dspf}.h5 -z 30 -Z 460"
source $scriptdir/pipelines/template_job_$template.sh

unset datastems
declare -a datastems
datastems[0]="$dataset"
export datastems

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=125000 wtime="01:00:00" q=""
for pf in '_maskDS' '_maskMM' '_maskMM-0.02' '_maskMB' '_ws_l0.99_u1.00_005'; do
    export jobname="rb_${dspf}${pf}"
    export cmd="python $scriptdir/convert/reduceblocks.py \
    $datadir datastem ${pf} 'stack' -d 1 ${ds} ${ds} -f 'np.amax' -o ${dspf}"
    source $scriptdir/pipelines/template_job_$template.sh
done

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="00:10:00" q="d"
# to get nifti's and h5 with same size as downsampled stack (1 px difference)
for pf in '' '_maskDS' '_maskMM' '_maskMM-0.02' '_maskMB'; do
for pf in '_ws_l0.99_u1.00_005' '_ws_l0.99_u1.00_005'; do
    export jobname="s2s_${dspf}${pf}"
    export cmd="python $scriptdir/convert/EM_stack2stack.py \
    $datadir/datastem${dspf}${pf}.h5 $datadir/datastem${dspf}${ds}${pf}.h5 \
    -e $ze $(echo $ye*$ds | bc) $(echo $xe*$ds | bc) -i 'zyx' -l 'zyx' \
    -X 1311 -Y 1255 -m"
    source $scriptdir/pipelines/template_job_$template.sh
    export jobname="nifti${pf}"
    export cmd="python $scriptdir/convert/EM_stack2stack.py \
    $datadir/datastem${dspf}${pf}.h5 $datadir/datastem${dspf}${ds}${pf}.nii.gz \
    -e $(echo $xe*$ds | bc) $(echo $ye*$ds | bc) $ze -i 'zyx' -l 'xyz' \
    -X 1311 -Y 1255 -m"
    source $scriptdir/pipelines/template_job_$template.sh
done
mv ${dataset}${dspf}${ds}_maskMM-0.h5 ${dataset}${dspf}${ds}_maskMM-0.02.h5

unset datastems
declare -a datastems
datastems[0]="$dataset"
export datastems
for pf in "${dspf}${ds}_ws_l0.99_u1.00_s005" "${dspf}${ds}_ws_l0.99_u1.00_s010"; do
    export jobname="s2s_${dspf}${pf}"
    export cmd="python $scriptdir/convert/EM_stack2stack.py \
    $datadir/datastem${pf}_mode.h5 $datadir/datastem${pf}.h5 \
    -e $ze $(echo $ye*$ds | bc) $(echo $xe*$ds | bc) -i 'zyx' -l 'zyx' \
    -X 1311 -Y 1255 -m"
    source $scriptdir/pipelines/template_job_$template.sh
    export jobname="nifti${pf}"
    export cmd="python $scriptdir/convert/EM_stack2stack.py \
    $datadir/datastem${pf}_mode.h5 $datadir/datastem${pf}.nii.gz \
    -e $(echo $xe*$ds | bc) $(echo $ye*$ds | bc) $ze -i 'zyx' -l 'xyz' \
    -X 1311 -Y 1255 -m"
    source $scriptdir/pipelines/template_job_$template.sh
done

###=========================================================================###
### 2D connected components in maskMM
###=========================================================================###
dspf='ds'; ds=7;
CC2Dstem='_labelMA_core2D'

unset datastems
declare -a datastems
datastems[0]=${dataset}${dspf}${ds}
export datastems


### basic 2D labeling
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""
export jobname="cc2D"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2D' \
-d 0 -o ${CC2Dstem} 'stack' -m"
source $scriptdir/pipelines/template_job_$template.sh


### apply criteria to create forward mappings
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""
export jobname="cc2Dfilter"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dfilter' \
-d 0 --maskMB '_maskMB' 'stack' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw" 'stack' \
-a 10 -A 1500 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh


### forward map
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""
export jobname="cc2Dprops"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dprops' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw" 'stack' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh


### 3D labeling
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""
export jobname="cc2Dto3D"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dto3Dlabel' \
-i "${CC2Dstem}_fw_label" 'stack' -o "${CC2Dstem}_fw_3Dlabeled" 'stack'"
source $scriptdir/pipelines/template_job_$template.sh


### merge neighbouring labels (on arcus, NOT arcus-b)
module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2
# produce pickled labelsets (1st iteration)
export template='single-pbs' additions='mpi'
export njobs=1 nodes=2 tasks=16 memcpu=60GB wtime="00:10:00" q="d"
export jobname="ccmerge1"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAstitch' -m \
-l "${CC2Dstem}_fw_3Dlabeled" 'stack' -o "${CC2Dstem}_fw_3Diter1" 'stack' \
-d 0 -q 2 -t 0.50"
source $scriptdir/pipelines/template_job_$template.sh
# produce stack from 1st iteration
export template='single-pbs' additions=''
export njobs=1 nodes=1 tasks=1 memcpu=60GB wtime="00:10:00" q="d"
export jobname="ccmerge2"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAfwmap' -d 0 \
-l "${CC2Dstem}_fw_3Dlabeled" 'stack' -o "${CC2Dstem}_fw_3Diter1" 'stack' \
--maskMM '_maskMM' 'stack' -c 6 1 1 -s 200 -r"
source $scriptdir/pipelines/template_job_$template.sh
###=========================================================================###
### proofreading MA
###=========================================================================###
# produce pickled labelsets (2nd iteration)
export template='single-pbs' additions='mpi'
export njobs=1 nodes=2 tasks=16 memcpu=60GB wtime="00:10:00" q="d"
export jobname="ccmerge3"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAstitch' -m \
-l "${CC2Dstem}_fw_3Diter1_proofread" 'stack' -o "${CC2Dstem}_fw_3Diter2" 'stack' \
-d 0 -q 2 -t 0.50"
source $scriptdir/pipelines/template_job_$template.sh
# produce stack from 2nd iteration
export template='single-pbs' additions=''
export njobs=1 nodes=1 tasks=1 memcpu=60GB wtime="00:10:00" q="d"
export jobname="ccmerge4"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAfwmap' -d 0 \
-l "${CC2Dstem}_fw_3Diter1_proofread" 'stack' -o "${CC2Dstem}_fw_3Diter2" 'stack' \
--maskMM '_maskMM' 'stack' -c 6 1 1"
source $scriptdir/pipelines/template_job_$template.sh
###=========================================================================###
### proofreading MA
###=========================================================================###
# produce pickled labelsets (3nd iteration)
export template='single-pbs' additions='mpi'
export njobs=1 nodes=2 tasks=16 memcpu=60GB wtime="00:10:00" q="d"
export jobname="ccmerge5"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAstitch' -m \
-l "${CC2Dstem}_fw_3Diter2" 'stack' -o "${CC2Dstem}_fw_3Diter3" 'stack' \
-d 0 -q 2 -t 0.50"
source $scriptdir/pipelines/template_job_$template.sh
# produce stack from 3nd iteration
export template='single-pbs' additions=''
export njobs=1 nodes=1 tasks=1 memcpu=60GB wtime="00:10:00" q="d"
export jobname="ccmerge6"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem -M 'MAfwmap' -d 0 \
-l "${CC2Dstem}_fw_3Diter2" 'stack' -o "${CC2Dstem}_fw_3Diter3" 'stack' \
--maskMM '_maskMM' 'stack' -r"  # is relabel a good idea???
source $scriptdir/pipelines/template_job_$template.sh
# clean tmp files
rm ${datastems[0]}${CC2Dstem}_fw*_host*_rank*.pickle


### inspect properties
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""
export jobname="ccinspect1"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dfilter' -E 1 \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw_nf" 'stack' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""
export jobname="ccinspect2"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dprops' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw_nf" 'stack' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh

### 3D CC
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""
export jobname="cc3D"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '3D' \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-a 200 -q 2000 -o '_labelMA_core3D' 'stack'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### proofreading MA
###=========================================================================###

# export template='array' additions='dellabels'
# export njobs=2 nodes=1 tasks=15 memcpu=3000 wtime="00:10:00" q="d"
# export jobname="dellabels"
# export cmd="python $scriptdir/supervoxels/delete_labels.py\
#  $datadir datastem\
#  -l '_labelMA_core' 'stack' -d deletelabels -o '_manedit'"
# source $scriptdir/pipelines/template_job_$template.sh



###=========================================================================###
### fill holes not reachable from unmyelinated axon space
###=========================================================================###
dspf='ds'; ds=7;
CC2Dstem='_labelMA_core2D'

unset datastems
declare -a datastems
datastems[0]=${dataset}${dspf}${ds}
export datastems


export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""
export jobname="fill_holes"
export cmd="python $scriptdir/supervoxels/fill_holes.py \
$datadir datastem -w 4 \
-L "${CC2Dstem}_fw_3Diter3" 'stack' \
--maskDS '_maskDS' 'stack' \
--maskMM '_maskMM' 'stack' \
--maskMX '_maskMM-0.02' 'stack' \
-o "_filled" 'stack'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### separate sheaths ds7
###=========================================================================###
dspf='ds'; ds=7;
# l="_labelMA_t0.1_ws_l0.99_u1.00_s010"
# l="_labelMA_core2D_fw_3Diter3_filled_manedit_proofread"
l='_labelMA_2D_proofread'

unset datastems
declare -a datastems
datastems[0]=${dataset}${dspf}${ds}
export datastems

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""
export jobname="sheaths"
export cmd="python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir datastem -o '_labelMM' 'stack' \
-l "${l}" 'stack' \
--maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack' \
--MAdilation 5"
source $scriptdir/pipelines/template_job_$template.sh

# export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
# export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="10:00:00" q=""
# export jobname="sheaths"
# export cmd="python $scriptdir/mesh/EM_separate_sheaths.py \
# $datadir datastem -o '_labelMM' 'stack' -w \
# -l '_labelMA_core2D_fw_3Dmerged' 'stack' \
# --maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack' > $datadir/datastem_labelMM.log"
# source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### agglomerate svox in full res image
###=========================================================================###
source datastems_90blocks.sh

svoxpf='_ws_l0.99_u1.00_s010'
dspf='ds'; ds=7;
lvol='_labelMA_2D_proofread'

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=60000 wtime="01:00:00" q=""
export jobname="agglo"
export cmd="python $scriptdir/supervoxels/agglo_from_labelsets.py \
$datadir datastem \
-s ${svoxpf} 'stack' -f ${svoxpf}_mode2 \
-l $datadir/${dataset}${dspf}${ds}_labelMA_t0.1_final_ws_l0.99_u1.00_s010_svoxsets.txt \
-o "_svoxsets" 'stack'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### Neuroproof ###
###=========================================================================###
CONDA_PATH="$(conda info --root)"
PREFIX="${CONDA_PATH}/envs/neuroproof-test"
NPdir="${HOME}/workspace/Neuroproof_minimal"
datadir="${DATA}/EM/Neuroproof/M3_S1_GNU_NP" && cd $datadir


###=========================================================================###
### watershed on prob_ics
###=========================================================================###
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/Neuroproof/M3_S1_GNU_NP/train" && cd $datadir
dataset="m000"
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"

unset datastems
declare -a datastems
datastems[0]=${datastem}
export datastems

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=125000 wtime="05:00:00" q=""

l=0.99; u=1.00; s=010;
svoxpf="_ws_l${l}_u${u}_${s}"
export jobname="ws${svoxpf}"
export cmd="python $scriptdir/supervoxels/EM_watershed.py \
$datadir datastem \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-p '_probs' 'volume/predictions' -c 1 \
-l $l -u $u -s $s -o ${svoxpf}"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### training
###=========================================================================###
module load hdf5-parallel/1.8.14_mvapich2_intel
module load mpi4py/1.3.1
module load python/2.7__gcc-4.8

trainset='train/m000_01000-01500_01000-01500_00030-00460'
strtype=2
iter=5
svoxpf='_ws_l0.99_u1.00_s010'
gtpf='_PA'
cltype='h5'  # 'h5'  # 'xml' #
classifier="_NPminimal${svoxpf}${gtpf}_str${strtype}_iter${iter}_parallel"

export template='single' additions='neuroproof-mpi' CONDA_ENV="neuroproof-test"
export njobs=1 nodes=1 tasks=16 memcpu=125000 wtime="10:00:00" q=""
export jobname="NP-train"
export cmd="$NPdir/NeuroProof_stack_learn \
-watershed $datadir/${trainset}${svoxpf}.h5 stack \
-prediction $datadir/${trainset}_probs.h5 volume/predictions \
-groundtruth $datadir/${trainset}${gtpf}.h5 stack \
-classifier $datadir/${trainset}${classifier}.${cltype} \
-iteration ${iter} -strategy ${strtype} -nomito"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### Neuroproof agglomeration
###=========================================================================###
module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

NPdir=${HOME}/workspace/Neuroproof_minimal
trainset="m000_01000-01500_01000-01500_00030-00460"
svoxpf='_ws_l0.99_u1.00_s010_svoxsets_MAdel'
classifier="_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallel"; cltype='h5';
thr=0.5; alg=1;
# cp ${DATA}/EM/Neuroproof/M3_S1_GNU_NP/train/${trainset}${classifier}.${cltype} .

export template='array' additions='neuroproof' CONDA_ENV="neuroproof-test"
export njobs=90 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""
export jobname="NP-test"
export cmd="$NPdir/NeuroProof_stack\
 -watershed $datadir/datastem${svoxpf}.h5 'stack'\
 -prediction $datadir/datastem_probs.h5 'volume/predictions'\
 -output $datadir/datastem${svopxpf}_prediction${classifier}${cltype}_thr${thr}_alg${alg}M.h5 'stack'\
 -classifier $datadir/${trainset}${classifier}.${cltype}\
 -threshold ${thr} -algorithm ${alg}"
source $scriptdir/pipelines/template_job_$template.sh


# agglo of all svox including MA
NPdir=${HOME}/workspace/Neuroproof_minimal
trainset="m000_01000-01500_01000-01500_00030-00460"
svoxpf='_ws_l0.99_u1.00_s010'
classifier="_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallel"; cltype='h5';
thr=0.5; alg=1;
# cp ${DATA}/EM/Neuroproof/M3_S1_GNU_NP/train/${trainset}${classifier}.${cltype} .
export template='array' additions='neuroproof' CONDA_ENV="neuroproof-test"
export njobs=45 nodes=1 tasks=2 memcpu=125000 wtime="02:00:00" q=""
export jobname="NP-test"
export cmd="$NPdir/NeuroProof_stack\
 -watershed $datadir/datastem${svoxpf}.h5 'stack'\
 -prediction $datadir/datastem_probs.h5 'volume/predictions'\
 -output $datadir/datastem${svoxpf}_prediction${classifier}${cltype}_thr${thr}_alg${alg}M.h5 'stack'\
 -classifier $datadir/${trainset}${classifier}.${cltype}\
 -threshold ${thr} -algorithm ${alg}"
source $scriptdir/pipelines/template_job_$template.sh



dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60GB wtime="10:00:00" q=""
# pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1'
# pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1M'
# pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.3_alg1'
pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.3_alg1'
export jobname="merge${pf}_mode"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' -o "${pf}_mode" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -n -r -F -B 1 ${ds} ${ds} -f 'mode'"
source $scriptdir/pipelines/template_job_$template.sh

pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1M'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir/pred_all ${datastems[*]} \
-i ${pf} 'stack' -o "${pf}_amax" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -n -F -B 1 ${ds} ${ds} -f 'np.amax' &


# on arcus with mpi: NO! neighbourmerge is not MPI aware!!
module load python/2.7
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2
dspf='ds'; ds=7;
export template='single-pbs' additions='mpi'
export njobs=1 nodes=6 tasks=5 memcpu=64GB wtime="02:10:00" q=""
export njobs=1 nodes=2 tasks=15 memcpu=50GB wtime="00:10:00" q="d"
pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1'
export jobname="merge${pf}_mpi2"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir/pred_all ${datastems[*]} \
-i ${pf} 'stack' -o "${pf}_mode_mpi" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -n -F -B 1 ${ds} ${ds} -f 'mode' -m"
source $scriptdir/pipelines/template_job_$template.sh
# mv ${dataset}${pf}.h5 ${dataset}${dspf}${ds}${pf}.h5




























###=========================================================================###
### 2D connected components in maskMM
###=========================================================================###
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="20:00:00" q=""

unset datastems
declare -a datastems
datastems[0]=$dataset
export datastems

# basic 2D labeling
export jobname="cc2D"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -d '2D' -o '_labelMA_core2D'"
source $scriptdir/pipelines/template_job_$template.sh
# apply criteria to create forward mappings  (FIXME: maskMB was not correct when running this!)
export jobname="cc2Dfilter"
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=3000 wtime="10:00:00" q=""
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dfilter' \
-i '_labelMA_core2D' -o '_labelMA_core2D_fw' \
-a 200 -A 30000 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh
### in 9 blocks:
source datastems_09blocks.sh
# split
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=${#datastems[@]} nodes=1 tasks=1 memcpu=40000 wtime="00:10:00" q="d"
for pf in '_labelMA_core2D' '_maskDS' '_maskMM' '_maskMB'; do
    export jobname="split_9b${pf}"
    export cmd="python $scriptdir/convert/EM_stack2stack.py \
    $datadir/${dataset}${pf}.h5 $datadir/datastem${pf}.h5 \
    -f 'stack' -g 'stack' -s 20 20 20 -b 0 0 30 -i zyx -l zyx -p datastem"
    source $scriptdir/pipelines/template_job_$template.sh
done
# forward map
export jobname="cc2Dprops"  # 40GB - 20GB
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=5 nodes=1 tasks=2 memcpu=125000 wtime="01:00:00" q=""
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dprops' \
-i '_labelMA_core2D' -o '_labelMA_core2D_fw' -b ${dataset} \
-p 'label' 'area' 'mean_intensity' 'eccentricity' 'solidity' 'extent' 'euler_number'"
source $scriptdir/pipelines/template_job_$template.sh
# label 3D
export jobname="cc2Dto3Dlabel"  # 40GB - 20GB
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=${#datastems[@]} nodes=1 tasks=1 memcpu=125000 wtime="05:00:00" q=""
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -M '2Dto3Dlabel' -o '_labelMA_core2D_fw'"
source $scriptdir/pipelines/template_job_$template.sh
# propagate and close
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=${#datastems[@]} nodes=1 tasks=1 memcpu=125000 wtime="10:00:00" q=""
export jobname="prop_9b"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem \
-l '_labelMA_core2D_fw_3Dlabeled' 'stack' \
-m 4 -t 0.50 -s 10000 -o '_labelMA_core2D_3Dmerged'"
source $scriptdir/pipelines/template_job_$template.sh

# mergeblocks (with overlapping labels)
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=125000 wtime="05:30:00" q=""
pf='_labelMA_core2D_labeled'; field='stack'
pf='_labelMA_core2D_merged'; field='stack'
pf='_labelMA_core2D_fw_3Dlabeled'; field='stack'
pf='_labelMA_core2D_fw_3Dmerged'; field='stack'
export jobname="merge_9b"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir $datadir/${dataset}${pf}.h5 \
-i ${datastems[*]} -t ${pf} -f $field -l 'zyx' \
-b 0 0 30 -p $xs $ys $zs -q $xm $ym $zm -s $xmax $ymax $zmax -r -n"
source $scriptdir/pipelines/template_job_$template.sh

# export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
# export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime="00:10:00" q="d"
# export jobname="nifti"
# pf='_labelMA_core2D_labeled'
# pf='_labelMA_core2D_merged'
# export cmd="python $scriptdir/convert/EM_stack2stack.py\
#  $datadir/$dataset${pf}.h5 $datadir/$dataset${pf}.nii.gz\
#  -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'\
#  -x 2850 -X 3850 -y 5950 -Y 6950"
# source $scriptdir/pipelines/template_job_$template.sh
pf=_labelMA_core2D_fw_3Dlabeled
python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNU_00000-03300_00000-03300_00030-00460${pf}.h5 \
$datadir/M3S1GNU_01500-02500_01500-02500_00030-00460${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' \
-x 1500 -X 2500 -y 1500 -Y 2500

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=125000 wtime="01:00:00" q=""
pf='_maskMB'
export jobname="nifti${pf}"
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 \
$datadir/datastem${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### connected components in maskMM-0.02 (TODO: in larger blocks??)
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=18 nodes=1 tasks=5 memcpu=12000 wtime="05:00:00" q=""

export jobname="conncomp3D"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem \
--maskMM '_maskMM-0.02' 'stack' -o '_labelMA_core3D' -d '3D'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### watershed on prob_ics
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=30 nodes=1 tasks=3 memcpu=125000 wtime="10:00:00" q=""

l=0.99; u=1.00; s=010;
svoxpf="_ws_l${l}_u${u}_${s}"
export jobname="ws${svoxpf}"
export cmd="python $scriptdir/supervoxels/EM_watershed.py \
$datadir datastem \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-p '_probs' 'volume/predictions' -c 1 \
-l $l -u $u -s $s -o ${svoxpf}"
source $scriptdir/pipelines/template_job_$template.sh

# datastem=M3S1GNU_01950-03050_02950-04050_00030-00460
# datastem=M3S1GNU_02950-04050_00950-02050_00030-00460
# datastem=M3S1GNU_02950-04050_01950-03050_00030-00460
datastem=M3S1GNU_02950-04050_02950-04050_00030-00460
datastems[4]=M3S1GNU_02950-04050_04950-06050_00030-00460
datastems[5]=M3S1GNU_03950-05050_00950-02050_00030-00460
datastems[6]=M3S1GNU_03950-05050_01950-03050_00030-00460
datastems[7]=M3S1GNU_05950-07050_04950-06050_00030-00460
datastems[8]=M3S1GNU_07950-09050_00950-02050_00030-00460
datastems[9]=M3S1GNU_07950-09050_01950-03050_00030-00460
datastems[10]=M3S1GNU_07950-09050_02950-04050_00030-00460
datastems[11]=M3S1GNU_07950-09050_03950-05050_00030-00460

python $scriptdir/supervoxels/EM_watershed.py \
$datadir $datastem \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-p '_probs' 'volume/predictions' -c 1 \
-l $l -u $u -s $s -o ${svoxpf} &

###=========================================================================###
### agglomerate watershedMA ()
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="agglo${svoxpf}"
export cmd="python $scriptdir/supervoxels/agglo_from_labelmask.py \
$datadir datastem \
-l '_labelMA_core_manedit' 'stack' -s ${svoxpf} 'stack' \
-o '_labelMA' -m '_maskMA'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### proofreading MA
###=========================================================================###

export deletefile="$datadir/m000_labelMA_core2D_delete.txt"
echo "m000_00000-01050_00000-01050_00030-00460: " >> $deletefile
# 1 is in UA: delete
# 121 is in a myelin loop
# split 16, merge party with 120
# 288 only partly filled
# 287 only partly filled
# 329 is partly filled
# merge 61 with 329 and fill section in between
# 98 is in myelin loop: delete
