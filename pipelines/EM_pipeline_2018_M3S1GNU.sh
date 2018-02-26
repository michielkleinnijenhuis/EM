###=========================================================================###
### copy data from previous pipeline
###=========================================================================###
# cp /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/archive/M3_S1_GNU_old/M3S1GNU/M3S1GNU.h5 /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU
# cp /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/archive/M3_S1_GNU_old/datastems_90blocks.sh /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU/datastems_blocks.sh


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
basedir="${DATA}/EM/M3"
dataset='M3S1GNU'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 9197 Image Length: 8786


###=========================================================================###
### apply ilastik classifier
###=========================================================================###
rem_host='ndcn0180@arcus-b.arc.ox.ac.uk'
rem_datadir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU"
loc_datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/M3S1GNUds7"
fname='pixclass_8class.ilp'
rsync -Pazv $loc_datadir/$fname $rem_host:$rem_datadir
fname='m000_01000-01500_01000-01500_00030-00460.h5'
rsync -Pazv $loc_datadir/$fname $rem_host:$rem_datadir

dset='stack'
pixprob_trainingset="pixclass_8class"

### on the full stack
export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=16 memcpu=125000 wtime="99:00:00" q=""
export jobname="ilastikfull"
export cmd="export LAZYFLOW_THREADS=16; export LAZYFLOW_TOTAL_RAM_MB=110000;\
$ilastik --headless \
--preconvert_stacks \
--project=$datadir/$pixprob_trainingset.ilp \
--output_axis_order=zyxc \
--output_format='compressed hdf5' \
--output_filename_format=$datadir/${dataset}_probs.h5 \
--output_internal_path=volume/predictions \
$datadir/$dataset.h5/$dset"
source $scriptdir/pipelines/template_job_$template.sh

# correct element_size_um attribute
import os
import numpy as np
from wmem import utils
datadir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU"
h5dset = "M3S1GNU.h5/stack"
h5file_in, ds_in, elsize, axlab = utils.h5_load(os.path.join(datadir, h5dset))
filestem = "M3S1GNU_probs/volume/predictions"
h5file_out, ds_out, _, _ = utils.h5_load(os.path.join(datadir, h5dset))
elsize = np.append(ds_in.attrs['element_size_um'], 1)
utils.h5_write_attributes(ds_out, element_size_um=elsize)
h5file_in.close()
h5file_out.close()


###=========================================================================###
### split _probs in blocks_500
###=========================================================================###
# cp datastems_blocks.sh datastems_blocks_500.sh
# vi datastems_blocks_500.sh  # xs=500; ys=500; xm=20; ym=20;
# mkdir -p $datadir/blocks_500
source datastems_blocks_500.sh
source find_missing_datastems.sh '_probs' 'h5' ${datadir}/blocks_500/
nstems=${#datastems[@]}
tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions='conda' CONDA_ENV='root'
export nodes=1 memcpu=6000 wtime='01:10:00' q='h'
export jobname='split'
export cmd="python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset}_probs.h5/volume/predictions \
$datadir/blocks_500/datastem_probs.h5/volume/predictions \
-p datastem"
source $scriptdir/pipelines/template_job_$template.sh

for i in `seq 0 $((njobs-1))`; do
sed -i -e "s/node=$tasks/node=1/g" EM_${jobname}_$i.sh
sed -i -e "s/ &//g" EM_${jobname}_$i.sh
sed -i -e "s/wait//g" EM_${jobname}_$i.sh
sbatch EM_${jobname}_$i.sh
# sbatch -p devel EM_${jobname}_$i.sh
done

###=========================================================================###
### EED
###=========================================================================###
module load hdf5-parallel/1.8.17_mvapich2_gcc
module load matlab/R2015a

mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/wmem/EM_eed_simple.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
cd $datadir

# scontrol update jobid=1613080 partition=compute MinMemoryCPU=60000 TimeLimit=01:30:00

source datastems_blocks_500.sh
source find_missing_datastems.sh '_probs0_eed2' 'h5' ${datadir}/blocks_500/
nstems=${#datastems[@]}
echo $nstems
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions=''
export nodes=1 memcpu=125000 wtime='03:10:00' q=''
export jobname='eed0'
export cmd="$datadir/bin/EM_eed_simple \
'$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
'1' '50' '1' '1'"
source $scriptdir/pipelines/template_job_$template.sh

source datastems_blocks_500.sh
source find_missing_datastems.sh '_probs1_eed2' 'h5' ${datadir}/blocks_500/
nstems=${#datastems[@]}
echo $nstems
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions=''
export nodes=1 memcpu=125000 wtime='03:10:00' q=''
export jobname='eed1'
export cmd="$datadir/bin/EM_eed_simple \
'$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
'2' '50' '1' '1'"
source $scriptdir/pipelines/template_job_$template.sh

# source datastems_blocks_500.sh
# source find_missing_datastems.sh '_probs2_eed2' 'h5' ${datadir}/blocks_500/
# nstems=${#datastems[@]}
# echo $nstems
# export tasks=16
# export njobs=$(( ($nstems + tasks-1) / $tasks))
# export template='array' additions=''
# export nodes=1 memcpu=125000 wtime='03:10:00' q='h'
# export jobname='eed2'
# export cmd="$datadir/bin/EM_eed_simple \
# '$datadir/blocks_500' 'datastem_probs' '/volume/predictions' '/probs_eed' \
# '3' '50' '1' '1' > $datadir/blocks_500/datastem_probs2_$jobname.log"
# source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### correct matlab h5 after EED
###=========================================================================###

import os
import numpy as np
import h5py
from glob import glob

datadir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU"

filestem = "M3S1GNU"
dset0 = 'stack'
h5path = os.path.join(datadir, filestem + '.h5')
h5file0 = h5py.File(h5path, 'a')
ds0 = h5file0[dset0]

files = glob(os.path.join(datadir, 'blocks_500', '*eed2.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['probs_eed']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um'][:3]
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
    h5file1.close()

h5file0.close()


###=========================================================================###
### merge blocks
###=========================================================================###
source datastems_blocks_500.sh
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime='03:10:00' q=''

export jobname="mergeprobs"
infiles=()
for file in `ls $datadir/blocks_500/*00460_probs0_eed2.h5`; do
    infiles+=("${file}/probs_eed")
done
export cmd="python $scriptdir/wmem/mergeblocks.py \
"${infiles[@]}" $datadir/${dataset}_probs00_eed2.h5/probs_eed \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### maskDS, maskMM, maskMM-0.02, maskMB
###=========================================================================###
# on full vol (without parallel, but not memory-intensive)
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""

export jobname="maskDS"
blocksize=20  # h5dump -pH $datadir/${dataset}.h5 | grep CHUNKED
cmd=""
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export cmd+="python $scriptdir/wmem/prob2mask.py \
$datadir/${dataset}.h5/stack \
$datadir/${dataset}_masks.h5/maskDS \
-l 0 -u 10000000 -D $z $Z 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh

export jobname="maskMM"
blocksize=27  # h5dump -pH $datadir/${dataset}_probs0_eed2.h5 | grep CHUNKED
cmd=""
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export cmd+="python $scriptdir/wmem/prob2mask.py \
$datadir/${dataset}_probs0_eed2.h5/probs_eed \
$datadir/${dataset}_masks.h5/maskMM \
-l 0.2 -D $z $Z 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh

export jobname="maskMA"
blocksize=27  # h5dump -pH $datadir/${dataset}_probs1_eed2.h5 | grep CHUNKED
cmd=""
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export cmd+="python $scriptdir/wmem/prob2mask.py \
$datadir/${dataset}_probs1_eed2.h5/probs_eed \
$datadir/${dataset}_masks.h5/maskMA \
-l 0.2 -D $z $Z 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### blockreduce
###=========================================================================###

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""
export jobname="rb_${dspf}_data"
blocksize=20  # h5dump -pH $datadir/${dataset}.h5 | grep CHUNKED
cmd=""
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export cmd+="python $scriptdir/wmem/downsample_blockwise.py \
$datadir/${dataset}.h5/stack \
$datadir/${dataset}${dspf}${ds}.h5/data \
-B 1 ${ds} ${ds} -f 'np.mean' -D $z $Z 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh

pf='_probs'; dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=125000 wtime="10:00:00" q=""
export jobname="rb_${dspf}_probs"
blocksize=10
cmd=""
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export cmd+="python $scriptdir/wmem/downsample_blockwise.py \
$datadir/${dataset}${pf}.h5/volume/predictions \
$datadir/${dataset}${dspf}${ds}${pf}.h5/volume/predictions \
-B 1 ${ds} ${ds} 1 -f 'np.mean' -D $z $Z 1 0 0 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="02:00:00" q=""
export jobname="rb_${dspf}_probs0_eed2"
blocksize=27  # h5dump -pH $datadir/${dataset}_probs0_eed2.h5 | grep CHUNKED
cmd=""
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export cmd+="python $scriptdir/wmem/downsample_blockwise.py \
$datadir/${dataset}_probs0_eed2.h5/probs_eed \
$datadir/${dataset}${dspf}${ds}_probs0_eed2.h5/probs_eed \
-B 1 ${ds} ${ds} -f 'np.mean' -D $z $Z 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=1 memcpu=60000 wtime="01:10:00" q=""
blocksize=27  # h5dump -pH $datadir/${dataset}_masks.h5 | grep CHUNKED
cmd=""
for pf in 'maskMM'; do  # 'maskDS' 'maskMM' 'maskMA'
for z in `seq 0 $blocksize $zs`; do
Z=$((z+blocksize))
Z=$(( Z < zs ? Z : zs ))
export jobname="rb_${dspf}${pf}"
export cmd+="python $scriptdir/wmem/downsample_blockwise.py \
$datadir/${dataset}_masks.h5/$pf $datadir/${dataset}${dspf}${ds}_masks.h5/$pf \
-B 1 ${ds} ${ds} -f 'np.amax' -D $z $Z 1 0 0 1 0 0 1; "
done
source $scriptdir/pipelines/template_job_$template.sh
done
