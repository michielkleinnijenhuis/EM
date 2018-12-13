###=========================================================================###
### prepare environment
###=========================================================================###
scriptdir=$HOME/workspace/EM
source $scriptdir/pipelines/datasets.sh
source $scriptdir/pipelines/functions.sh
source $scriptdir/pipelines/submission.sh

compute_env='LOCAL'
prep_environment $scriptdir $compute_env

dataset='M3S1GNU'
dataset='B-NT-S9-2a'
# dataset='B-NT-S10-2d_ROI_00'
# dataset='B-NT-S10-2d_ROI_02'
# dataset='B-NT-S10-2f_ROI_00'
# dataset='B-NT-S10-2f_ROI_01'
# dataset='B-NT-S10-2f_ROI_02'
bs='0500'
prep_dataset $dataset $bs
echo ${#datastems[@]}

echo -n -e "\033]0;$dataset\007"

###=========================================================================###
### nifti's
###=========================================================================###

# datastem='B-NT-S10-2f_ROI_00_00000-00520_00000-00520_00000-00184'
# h52nii '' blocks_0500/$datastem '' 'data' '' '' '-u -i zyx -o xyz'
# h52nii '' blocks_0500/$datastem '_masks' 'maskDS' '' '' '-i zyx -o xyz'
# h52nii '' blocks_0500/$datastem '_masks' 'maskMM' '' '' '-i zyx -o xyz'

h52nii '' $dataset_ds '' 'data' '' '' '-u -i zyx -o xyz'

h52nii '' $dataset_ds '_probs_eed_sum0247' 'sum0247_eed' '' '' '-u -i zyx -o xyz'
h52nii '' $dataset_ds '_probs_eed_sum16' 'sum16_eed' '' '' '-u -i zyx -o xyz'
h52nii '' $dataset_ds '_probs_eed_probMA' 'probMA_eed' '' '' '-u -i zyx -o xyz'
# h52nii '' $dataset_ds '_probs1' 'probMA' '' '' '-u -i zyx -o xyz'

h52nii '' $dataset_ds '_masks' 'maskDS' '' '' '-i zyx -o xyz'

h52nii '' $dataset_ds '_masks_maskDS' 'maskDS' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_masks_maskMM' 'maskMM' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_masks_maskICS' 'maskICS' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_masks_maskMA' 'maskMA' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_masks_maskMA_ds-0.8' 'maskMA_ds-0.8' '' '' '-i zyx -o xyz'

# h52nii '' $dataset_ds '_labels' 'labelMA_core2D' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA' 'area' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset_ds '_labels_labelMA' 'eccentricity' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset_ds '_labels_labelMA' 'euler_number' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA' 'extent' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset_ds '_labels_labelMA' 'label' '' '' '-i zyx -o xyz -d uint32'
h52nii '' $dataset_ds '_labels_labelMA' 'mean_intensity' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset_ds '_labels_labelMA' 'solidity' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset_ds '_labels_labelMA_core2D' 'labelMA_core2D' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_pred' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Dlabeled' '' '' '-i zyx -o xyz'


h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Dlabeled_NoR' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Dlabeled_NoR_steps/boundarymask' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Dlabeled_NoR_steps/labels_NoR' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1_NoR_steps/largelabels' '' '' '-i zyx -o xyz -d uint16'


h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1' '' '' '-i zyx -o xyz -d uint16'


h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1_filled' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1_NoR_steps/boundarymask' '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1_NoR_steps/labels_NoR' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1_NoR_steps/largelabels' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA' 'labelMA_3Diter1_NoR_steps/filled' '' '' '-i zyx -o xyz -d uint16'

# h52nii '' blocks_2000/M3S1GNU_00000-02050_00000-02050_00000-00430 '_ws' 'l0.99_u1.00_s010' '' '' '-i zyx -o xyz -d uint16'
h52nii '' "blocks_2000/${dataset}_00000-02050_00000-02050_00000-00430" '_ws' 'l0.99_u1.00_s010' '' '' '-i zyx -o xyz -d uint16 -D 0 500 1 0 500 1 0 0 1'

for i in `seq 0 7`; do
    h52nii '' "blocks_0500/${dataset}_00000-00520_00000-00520_00000-00135" '_probs' 'volume/predictions' '' '' "-i zyxc -o xyzc -d uint16 -D 0 0 1 0 0 1 0 0 1 $i $((i+1)) 1"
done
h52nii '' "blocks_0500/${dataset}_00000-00520_00000-00520_00000-00135" '_probs' 'volume/predictions' '' '' "-u -i zyxc -o xyzc"

labelMA_3Diter1_NoR_steps/labels_NoR
labelMA_3Diter1_NoR_steps/largelabels



h52nii 'd' $dataset '' 'data' '' '' '-u -i zyx -o xyz -D 0 0 1 5000 5003 1 0 0 1'
h52nii 'd' $dataset '_masks' 'maskDS' '' '' '-i zyx -o xyz -D 0 0 1 5000 5003 1 0 0 1'
h52nii 'd' $dataset '_masks_maskDS' 'maskDS' '' '' '-i zyx -o xyz -D 0 0 1 5000 5003 1 0 0 1'
h52nii 'd' $dataset '_masks_maskMA' 'maskMA' '' '' '-i zyx -o xyz -D 0 0 1 5000 5003 1 0 0 1'


h52nii 'd' $dataset '' 'data' '' '' '-u -i zyx -o xyz -D 100 103 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_masks_maskMA' 'maskMA' '' '' '-i zyx -o xyz -D 100 103 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_probs_eed_sum0247' 'sum0247_eed' '' '' '-i zyx -o xyz -D 100 103 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_probs_eed_sum16' 'sum16_eed' '' '' '-i zyx -o xyz -D 100 103 1 0 0 1 0 0 1'


h52nii 'd' $dataset '_probs1' 'probMA' '' '' '-u -i zyx -o xyz -D 0 0 1 5000 5003 1 0 0 1'
h52nii 'd' $dataset_ds '' 'data' '' '' '-u -i zyx -o xyz -D 0 0 1 500 503 1 0 0 1'
h52nii 'd' $dataset '_masks' 'maskDS' '' '' '-i zyx -o xyz -D 0 0 1 5000 5003 1 0 0 1'
h52nii 'd' $dataset_ds '_masks' 'maskDS' '' '' '-i zyx -o xyz -D 0 0 1 500 503 1 0 0 1'



python $scriptdir/wmem/stack2stack.py $datadir/${dataset}_00000-00460.h5/data $datadir/${dataset}.h5/data -D 30 0 1 0 0 1 0 0 1
declare ipf='' ids='data' opf='' ods='data' \
    brfun='np.mean' brvol='' slab=14 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce '' '' $ids '' $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
declare ipf='' ids='data' opf='_masks' ods='maskDS' slab=16 arg='-g -l 0 -u 10000000'
prob2mask 'h' '' $ids $opf $ods $slab "$arg"
declare ipf='_masks' ids='maskDS' opf='_masks' ods='maskDS' \
    brfun='np.amax' brvol='' slab=16 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"

# python $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}_00000-00460.h5/data $datadir/${dataset_ds}.h5/data -D 30 0 1 0 0 1 0 0 1
# python $scriptdir/wmem/stack2stack.py $datadir/${dataset}_masks.h5/maskDS $datadir/${dataset}_masks_maskDS.h5/maskDS -D 30 0 1 0 0 1 0 0 1

dset='maskDS'
h5copy -p -i ${dataset_ds}_masks.h5 -s $dset -o ${dataset_ds}_masks_$dset.h5 -d $dset

dset='maskDS'
h5copy -p -i M3S1GNU_masks_$dset.h5 -s $dset -o M3S1GNU_masks.h5 -d $dset
for dset in 'maskMM' 'maskICS' 'maskMA'; do
    h5copy -p -i M3S1GNU_masks_tmp.h5 -s $dset -o M3S1GNU_masks.h5 -d $dset
done

h5copy -p -i ${dataset_ds}_probs_eed.h5 -s sum0247_eed -o ${dataset_ds}_probs_eed_sum0247.h5 -d sum0247_eed
h5copy -p -i ${dataset_ds}_probs_eed.h5 -s sum16_eed -o ${dataset_ds}_probs_eed_sum16.h5 -d sum16_eed

for dset in 'maskDS' 'maskMM' 'maskICS' 'maskMA'; do
    h5copy -p -i ${dataset_ds}_masks.h5 -s $dset -o ${dataset_ds}_masks_${dset}.h5 -d $dset
done
for dset in 'maskDS' 'maskMM' 'maskICS' 'maskMA'; do  # 'maskMM_raw'
    h5copy -p -i ${dataset}_masks.h5 -s $dset -o ${dataset}_masks_${dset}.h5 -d $dset
done


dset='maskDS'
h5copy -p -i tmp/M3S1GNUds7_masks.h5 -s $dset -o M3S1GNUds7_masks.h5 -d $dset

### dataset='B-NT-S10-2f_ROI_01' & dataset='B-NT-S10-2f_ROI_02'
for f in `ls *_probs1_eed.h5`; do
    h5copy -p -i $f -s probMA -o ${f}tmp -d probMA_eed
done
rm *_probs1_eed.h5
rename _probs1_eed.h5tmp _probs1_eed.h5 *_probs1_eed.h5tmp

for f in `ls *_probs0_eed.h5`; do
    h5copy -p -i $f -s probs_eed -o ${f}tmp -d probMM_eed
done
rm *_probs0_eed.h5
rename _probs0_eed.h5tmp _probs0_eed.h5 *_probs0_eed.h5tmp

for dset in 'sum0247' 'sum16'; do
    for f in `ls *_probs_sums.h5`; do
        h5copy -p -i $f -s $dset -o ${f/_sums/} -d $dset
    done
done
rm *_sums.h5

for f in `ls *_probs_eed.h5`; do
    h5copy -p -i $f -s 'sum0247_eed' -o ${f/_eed/_eed_sum0247} -d 'sum0247_eed'
    h5copy -p -i $f -s 'sum16_eed' -o ${f/_eed/_eed_sum16} -d 'sum16_eed'
done
# rm *_probs_eed.h5

# rm *_probs_eed_sum16.h5
for f in `ls *_probs_eed.h5`; do
    h5copy -p -i $f -s 'sum0247_eed' -o ${f/_eed/_eed_sum0247} -d 'sum0247_eed'
    h5copy -p -i $f -s 'sum16_eed' -o ${f/_eed/_eed_sum16} -d 'sum16_eed'
    # h5copy -p -i $f -s 'probMA' -o ${f/_eed/_eed_probMA} -d 'probMA_eed'
done
rm *_probs_eed.h5
for f in `ls *_probs_eed_sum16.h5`; do
    h5copy -p -i $f -s 'sum16_eed' -o ${f/_eed_sum16/_eed_sum16tmp} -d 'sum16_eed'
    h5copy -p -i $f -s 'probMA_eed' -o ${f/_eed_sum16/_eed_probMA} -d 'probMA_eed'
done

for f in `ls *_masks.h5`; do
    h5copy -p -i $f -s 'maskICS' -o ${f/_masks/_masks_maskICS} -d 'maskICS'
    h5copy -p -i $f -s 'maskMA' -o ${f/_masks/_masks_maskMA} -d 'maskMA'
    h5copy -p -i $f -s 'maskMM' -o ${f/_masks/_masks_maskMM} -d 'maskMM'
    h5copy -p -i $f -s 'maskMM_steps' -o ${f/_masks/_masks_maskMM} -d 'maskMM_steps'
done
rm *_masks.h5



h5copy -p -i ${dataset}_masks_maskMM.h5 -s maskMM -o ${dataset}_masks_maskMMtmp.h5 -d maskMM
h5copy -p -i ${dataset}_masks_maskMM.h5 -s maskMM_raw -o ${dataset}_masks_maskMM_raw.h5 -d maskMM_raw




unset files && declare -a files
files+=( B-NT-S10-2f_ROI_00_00000-00520_04980-05520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_00480-01020_05480-06020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_00480-01020_05480-06020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_00980-01520_05980-06520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_00980-01520_05980-06520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_01480-02020_06480-07020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_01980-02520_06980-07520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_02480-03020_07480-08020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_02980-03520_07980-08316_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_03980-04520_00000-00520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_04480-05020_00480-01020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_04980-05520_00980-01520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_05480-06020_01480-02020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_05980-06520_01980-02520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_06480-07020_02480-03020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_06980-07520_02980-03520_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_07480-08020_03480-04020_00000-00184_probs_eed.h5 )
files+=( B-NT-S10-2f_ROI_00_07980-08423_03980-04520_00000-00184_probs_eed.h5 )
for f in ${files[@]}; do
    h5copy -p -i $f -s 'sum0247_eed' -o ${f/_eed/_eed_sum0247} -d 'sum0247_eed'
    h5copy -p -i $f -s 'sum16_eed' -o ${f/_eed/_eed_sum16} -d 'sum16_eed'
done


# rm ${dataset}_?????-?????_?????-?????_00000-00???_probs.h5

declare ipf='_masks' ids='maskMA' opf='_masks' ods='maskMA'
unset infiles && declare -a infiles && get_infiles_datastems
infiles=( "${infiles[@]:0:162}" )
echo ${#infiles[@]}
mergeblocks 'h' '' '0' $ipf $ids $opf $ods ''









CONDA_PATH="$(conda info --root)"
PREFIX="${CONDA_PATH}/envs/neuroproof-test"
NPdir="${HOME}/workspace/Neuroproof_minimal"

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


neuroproof_graph_learn
neuroproof_graph_predict
neuroproof_graph_analyze
neuroproof_stack_viwer
basic_rag_test  basic_stack_test  neuroproof_create_spgraph  neuroproof_graph_analyze  neuroproof_graph_analyze_gt  neuroproof_graph_build_stream  neuroproof_graph_learn  neuroproof_graph_predict  neuroproof_stack_viewer
