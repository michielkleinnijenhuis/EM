### in 90 blocks
source datastems_90blocks.sh
# split
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"
pf='_labelMA_core2D'
export jobname="split_90b"
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}${pf}.h5 $datadir/datastem${pf}.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -b 0 0 30 -i zyx -l zyx -p datastem"
source $scriptdir/pipelines/template_job_$template.sh
# label
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=6 nodes=1 tasks=15 memcpu=125000 wtime="01:00:00" q=""
export jobname="conncomp_90b"
export cmd="python $scriptdir/supervoxels/conn_comp.py \
$datadir datastem -o '_labelMA_core2D'"
source $scriptdir/pipelines/template_job_$template.sh
# propagate and close
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=125000 wtime="01:00:00" q=""
# export njobs=1 nodes=1 tasks=5 memcpu=60000 wtime="01:00:00" q="d"
export jobname="prop"
export cmd="python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir datastem \
-l '_labelMA_core2D_labeled' 'stack' --maskMM '_maskMM' 'stack' \
-m 4 -t 0.01 -s 10000 -o '_labelMA_core2D_merged'"
source $scriptdir/pipelines/template_job_$template.sh
# to nifti
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=9 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"
export jobname="nifti"
pf='_labelMA_core2D_merged'
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'"
source $scriptdir/pipelines/template_job_$template.sh
# mergeblocks (with overlapping labels)
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=50000 wtime="02:30:00" q=""
pf='_labelMA_core2D_labeled'; field='stack'
pf='_labelMA_core2D_merged'; field='stack'
export jobname="merge_90b"
export cmd="python $scriptdir/convert/EM_mergeblocks.py \
$datadir $datadir/${dataset}${pf}_90b.h5 \
-i ${datastems[*]} -t ${pf} -f $field -l 'zyx' \
-b 0 0 30 -p $xs $ys $zs -q $xm $ym $zm -s $xmax $ymax $zmax -r -n"
source $scriptdir/pipelines/template_job_$template.sh

# export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
# export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime="00:10:00" q="d"
# export jobname="nifti"
# pf='_labelMA_core2D_labeled_90b'
# export cmd="python $scriptdir/convert/EM_stack2stack.py\
#  $datadir/$dataset${pf}.h5 $datadir/$dataset${pf}.nii.gz\
#  -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'\
#  -x 1500 -X 2500 -y 1500 -Y 2500"
# source $scriptdir/pipelines/template_job_$template.sh
#
# M3S1GNU_labelMA_core2D_labeled_90b.h5
