export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"

source activate root

f=m000_maskMM
f=m000_03000-04000_03000-04000_00030-00460
h5repack -v -f GZIP=4 $f.h5 ${f}_gzip.h5 &




cd ~/Library/Application\ Support/Blender/2.78/scripts/addons/

'/Users/michielk/workspace/blender-build/build_darwin/bin/blender.app/Contents/Resources/2.78/scripts/addons/mesh_extra_tools/__init__.py'



blender -b -P /home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/mesh/stl2blender.py -- /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1 _labelUA_final_d0.02collapse_s100-0.1_1 -L _labelUA_final -S /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03919.01.stl /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03920.01.stl /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03922.01.stl /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03923.01.stl /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03924.01.stl /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03925.01.stl /data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1/_labelUA_final.03926.01.stl  -s 100 0.1 True True True -d 0.02 -e 0


host="ndcn0180@arcus-b.arc.ox.ac.uk"
remdir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1"
localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/M3S1GNUds7/dmcsurf_1-1-1"
rsync -avz "$host:$remdir/_labelUA_final.33???.01.stl" $localdir

host="ndcn0180@arcus-b.arc.ox.ac.uk"
remdir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new"

f="M3S1GNU_*_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1_amaxX.npy"
f="M3S1GNU_00000-01050_*-*_00030-00460_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1.h5"
rsync -avz "$host:$remdir/${f}" $localdir




##
export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/M3/M3_S1_GNU_old" && cd $datadir
source datastems_90blocks.sh

dspf='ds'; ds=7;
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60GB wtime="10:00:00" q=""
pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1'
export jobname="merge${pf}_amax"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' -o "${pf}_amaxY" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -n -r -F -B 1 ${ds} ${ds} -f 'np.amax'"
source $scriptdir/pipelines/template_job_$template.sh



import os
import h5py
import numpy as np
from skimage.measure import regionprops, label

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new"
dset_name = 'M3S1GNU'

labelvolume = ['_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1_amaxX', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :]
f.close()

## label2stl
datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new"
datastem='M3S1GNUds7'
python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L '_labelUA' -f 'stack' -o 30 0 0


export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/M3/M3_S1_GNU_old" && cd $datadir

datastem='M3S1GNUds7'
export template='single' additions='' CONDA_ENV=''
export njobs=1 nodes=1 tasks=1 memcpu=50000 wtime="100:00:00" q=""
export jobname="label2stl"
export cmd="python $scriptdir/mesh/label2stl.py $datadir/ds7_arc $datastem \
-L '_labelUA_final' -f 'stack' -o 30 0 0"
source $scriptdir/pipelines/template_job_$template.sh






### merge probs
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=60GB wtime="10:00:00" q=""
pf='_probs'
export jobname="merge${pf}"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'volume/predictions' -o ${pf}_merged 'volume/predictions' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax -m"
source $scriptdir/pipelines/template_job_$template.sh
