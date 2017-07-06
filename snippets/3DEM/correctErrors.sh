### set the elsize and axislabels in stacks that miss them
for f in `ls *_probs.h5`; do
h5ls -v $f/volume
done

for f in `ls *_alg1.h5`; do
h5ls $f
done


import os
import h5py
import glob

inputdir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
inputdir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/pred_new"
inputdir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
inputdir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
inputdir = '/Users/michielk/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/M3S1GNUvols/data'

regexp = "*_probs.h5"
elsize = [0.05, 0.0073, 0.0073, 1]
axislabels = 'zyxc'
field = 'volume/predictions'

regexp = "M3S1GNUds7_labelMA_core*.h5"
regexp = "M3S1GNUds7_*_final.h5"
regexp = "M3S1GNUds7_labelMA_2D*.h5"
elsize = [0.05, 0.0511, 0.0511]
axislabels = 'zyx'
field = 'stack'

regexp = "*_labelMA_core2D.h5"
regexp = "*_labelMA_core2D_merged.h5"
regexp = "*_ws_l0.99_u1.00_s010_svoxsets_agglo.h5"
regexp = "*_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1.h5"
regexp = "*_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1M.h5"
elsize = [0.05, 0.0073, 0.0073]
axislabels = 'zyx'
field = 'stack'


infiles = glob.glob(os.path.join(inputdir, regexp))
for fname in infiles:
    try:
        f = h5py.File(fname, 'a')
        f[field].attrs['element_size_um'] = elsize
        for i, l in enumerate(axislabels):
            f[field].dims[i].label = l
        f.close()
        print("%s done" % fname)
    except:
        print(fname)


### move and process backups of some _probs.h5 file that were missing
unset datastems
declare -a datastems
datastems[0]=M3S1GNU_00000-01050_01950-03050_00030-00460
datastems[1]=M3S1GNU_00000-01050_02950-04050_00030-00460
datastems[2]=M3S1GNU_00000-01050_04950-06050_00030-00460
datastems[3]=M3S1GNU_00000-01050_05950-07050_00030-00460
datastems[4]=M3S1GNU_01950-03050_02950-04050_00030-00460
datastems[5]=M3S1GNU_02950-04050_00950-02050_00030-00460
datastems[6]=M3S1GNU_02950-04050_01950-03050_00030-00460
datastems[7]=M3S1GNU_02950-04050_02950-04050_00030-00460
datastems[8]=M3S1GNU_02950-04050_04950-06050_00030-00460
datastems[9]=M3S1GNU_03950-05050_00950-02050_00030-00460
datastems[10]=M3S1GNU_03950-05050_01950-03050_00030-00460
datastems[11]=M3S1GNU_05950-07050_04950-06050_00030-00460
datastems[12]=M3S1GNU_07950-09050_00950-02050_00030-00460
datastems[13]=M3S1GNU_07950-09050_01950-03050_00030-00460
datastems[14]=M3S1GNU_07950-09050_02950-04050_00030-00460
datastems[15]=M3S1GNU_07950-09050_03950-05050_00030-00460
export datastems

for d in ${datastems[@]}; do
    cp probs_backup/${d}_probs.h5 .
done

ls -l *_probs.h5

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=8 nodes=1 tasks=2 memcpu=125000 wtime="03:00:00" q=""
export jobname="zyxc_p"
pf='_probs'
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 $datadir/datastem${pf}_zyxc.h5 \
-e 0.05 0.0073 0.0073 1 -i 'xyzc' -l 'zyxc' \
-f 'volume/predictions'  -g 'volume/predictions'"
source $scriptdir/pipelines/template_job_$template.sh

ls -l *_probs.h5

for d in ${datastems[@]}; do
    mv ${d}_probs_zyxc.h5 ${d}_probs.h5
done


# failed ws_l0.99_u1.00_s005.h5
unset datastems
declare -a datastems
datastems[0]=M3S1GNU_01950-03050_02950-04050_00030-00460
datastems[1]=M3S1GNU_02950-04050_00950-02050_00030-00460
datastems[2]=M3S1GNU_02950-04050_01950-03050_00030-00460
datastems[3]=M3S1GNU_02950-04050_02950-04050_00030-00460
datastems[4]=M3S1GNU_02950-04050_04950-06050_00030-00460
datastems[5]=M3S1GNU_03950-05050_00950-02050_00030-00460
datastems[6]=M3S1GNU_03950-05050_01950-03050_00030-00460
datastems[7]=M3S1GNU_05950-07050_04950-06050_00030-00460
datastems[8]=M3S1GNU_07950-09050_00950-02050_00030-00460
datastems[9]=M3S1GNU_07950-09050_01950-03050_00030-00460
datastems[10]=M3S1GNU_07950-09050_02950-04050_00030-00460
datastems[11]=M3S1GNU_07950-09050_03950-05050_00030-00460
export datastems

unset datastems
declare -a datastems
datastems[0]=M3S1GNU_02950-04050_01950-03050_00030-00460
datastems[1]=M3S1GNU_07950-09050_03950-05050_00030-00460
export datastems

unset datastems
declare -a datastems
datastems[0]=M3S1GNU_00000-01050_01950-03050_00030-00460
datastems[1]=M3S1GNU_00000-01050_02950-04050_00030-00460
datastems[2]=M3S1GNU_00000-01050_04950-06050_00030-00460
datastems[3]=M3S1GNU_00000-01050_05950-07050_00030-00460
export datastems

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=2 nodes=1 tasks=2 memcpu=125000 wtime="05:00:00" q=""
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="ws${svoxpf}"
export cmd="python $scriptdir/supervoxels/EM_watershed.py \
$datadir datastem \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-p '_probs' 'volume/predictions' -c 1 \
-l $l -u $u -s $s -o ${svoxpf}"
source $scriptdir/pipelines/template_job_$template.sh






## some watershed results were missing:
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=11 nodes=1 tasks=3 memcpu=125000 wtime="04:00:00" q=""
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_${s}"
source datastems_90blocks.sh
source find_missing_datastems.sh $svoxpf 'h5'
export jobname="ws${svoxpf}"
export cmd="python $scriptdir/supervoxels/EM_watershed.py \
$datadir datastem \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-p '_probs' 'volume/predictions' -c 1 \
-l $l -u $u -s $s -o ${svoxpf}"
source $scriptdir/pipelines/template_job_$template.sh


source find_missing_datastems.sh '_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1M' 'h5'

M3S1GNU_04950-06050_05950-07050_00030-00460
M3S1GNU_05950-07050_03950-05050_00030-00460
M3S1GNU_05950-07050_05950-07050_00030-00460
M3S1GNU_06950-08050_00000-01050_00030-00460
M3S1GNU_06950-08050_00950-02050_00030-00460
M3S1GNU_06950-08050_01950-03050_00030-00460
M3S1GNU_06950-08050_02950-04050_00030-00460
M3S1GNU_06950-08050_03950-05050_00030-00460
M3S1GNU_06950-08050_04950-06050_00030-00460
M3S1GNU_06950-08050_05950-07050_00030-00460
M3S1GNU_06950-08050_06950-08050_00030-00460
M3S1GNU_06950-08050_07950-08786_00030-00460
M3S1GNU_07950-09050_00000-01050_00030-00460
M3S1GNU_07950-09050_00950-02050_00030-00460
M3S1GNU_07950-09050_01950-03050_00030-00460
M3S1GNU_07950-09050_02950-04050_00030-00460
M3S1GNU_07950-09050_03950-05050_00030-00460
M3S1GNU_07950-09050_04950-06050_00030-00460
M3S1GNU_07950-09050_05950-07050_00030-00460
M3S1GNU_07950-09050_06950-08050_00030-00460
M3S1GNU_07950-09050_07950-08786_00030-00460

module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

NPdir=${HOME}/workspace/Neuroproof_minimal
trainset="m000_01000-01500_01000-01500_00030-00460"
svoxpf='_ws_l0.99_u1.00_s010_svoxsets_MAdel'
classifier="_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallel"; cltype='h5';
thr=0.5; alg=1;
# cp ${DATA}/EM/Neuroproof/M3_S1_GNU_NP/train/${trainset}${classifier}.${cltype} .

export template='array' additions='neuroproof' CONDA_ENV="neuroproof-test"
export njobs=1 nodes=1 tasks=1 memcpu=125000 wtime="05:00:00" q=""
export jobname="NP-test"
export cmd="$NPdir/NeuroProof_stack\
 -watershed $datadir/datastem${svoxpf}.h5 'stack'\
 -prediction $datadir/datastem_probs.h5 'volume/predictions'\
 -output $datadir/datastem${svopxpf}_prediction${classifier}${cltype}_thr${thr}_alg${alg}M.h5 'stack'\
 -classifier $datadir/${trainset}${classifier}.${cltype}\
 -threshold ${thr} -algorithm ${alg}"
source $scriptdir/pipelines/template_job_$template.sh

source find_missing_datastems.sh '_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1M' 'h5'

unset datastems
declare -a datastems
datastems[0]=M3S1GNU_06950-08050_02950-04050_00030-00460
export datastems






export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
export PATH=$CONDA_PATH:$PATH
export CONDA_ENV=neuroproof-test
source activate $CONDA_ENV
export LD_LIBRARY_PATH=$CONDA_PATH/envs/$CONDA_ENV/lib

module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

NPdir=${HOME}/workspace/Neuroproof_minimal
trainset="m000_01000-01500_01000-01500_00030-00460"
svoxpf='_ws_l0.99_u1.00_s010_svoxsets_MAdel'
classifier="_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallel"; cltype='h5';
thr=0.5; alg=1;

datastem=M3S1GNU_06950-08050_02950-04050_00030-00460
datastem=M3S1GNU_07950-09050_04950-06050_00030-00460
datastem=M3S1GNU_07950-09050_05950-07050_00030-00460
datastem=M3S1GNU_07950-09050_06950-08050_00030-00460
datastem=M3S1GNU_07950-09050_07950-08786_00030-00460
$NPdir/NeuroProof_stack\
 -watershed $datadir/${datastem}${svoxpf}.h5 'stack'\
 -prediction $datadir/${datastem}_probs.h5 'volume/predictions'\
 -output $datadir/${datastem}${svopxpf}_prediction${classifier}${cltype}_thr${thr}_alg${alg}_tmpM.h5 'stack'\
 -classifier $datadir/${trainset}${classifier}.${cltype}\
 -threshold ${thr} -algorithm ${alg}
