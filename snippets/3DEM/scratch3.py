# deletion accident
pf='_probs1_eed2.h5'
pf='_probs0_eed2.h5'
pf='_probs1_eed2_zyx.h5'
pf='_probs_zyxc.h5/volume'
for i in `seq 0 89`; do
    h5ls ${datastems[i]}${pf}
done

for i in `seq 40 89`; do
    unset datastems[i]
done

for i in `seq 0 39`; do
    mv ${datastems[i]}_probs0_eed2.h5 probs0_eed2_faulty/
done


import os
import numpy as np
datadir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
fname = 'M3S1GNU_00000-03300_00000-03300_00030-00460_labelMA_core2D_fw.npy'
fname = 'M3S1GNU_00000-03300_00000-03300_00030-00460_labelMA_core2D_labeled_test.npy'
fname = 'M3S1GNU_00000-03300_02900-06400_00030-00460_labelMA_core2D_labeled_test.npy'
fname = 'M3S1GNU_02900-06400_00000-03300_00030-00460_labelMA_core2D_labeled_test.npy'
fw = np.load(os.path.join(datadir, fname))


### testing labelmerge
export dataset="M3S1GNU"
xmax=9179; ymax=8786; zmax=460;
xs=1000; ys=1000; zs=430;
xm=50; ym=50; zm=0;
z=30; Z=460;
unset datastems
declare -a datastems
i=0
for x in `seq 0 $xs 0`; do
    [ $x == 9000 ] && X=$xmax || X=$((x+xs+xm))
    [ $x == 0 ] || x=$((x-xm))
    for y in `seq 0 $ys $ymax`; do
        [ $y == 8000 ] && Y=$ymax || Y=$((y+ys+ym))
        [ $y == 0 ] || y=$((y-ym))
        xrange=`printf %05d ${x}`-`printf %05d ${X}`
        yrange=`printf %05d ${y}`-`printf %05d ${Y}`
        zrange=`printf %05d ${z}`-`printf %05d ${Z}`
        echo ${dataset}_${xrange}_${yrange}_${zrange}
        datastems[$i]=${dataset}_${xrange}_${yrange}_${zrange}
        i=$((i+1))
    done
done
export datastems

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=50000 wtime="02:30:00" q="d"
pf='_labelMA_core2D_labeled'; field='stack'
export jobname="merge"
export cmd="python $scriptdir/convert/EM_mergeblocks.py\
 $datadir $datadir/${dataset}${pf}_test.h5\
 -i ${datastems[*]} -t ${pf} -f $field -l 'zyx'\
 -b 0 0 30 -p $xs $ys $zs -q $xm $ym $zm -s $xmax $ymax $zmax -r -n"
source $scriptdir/pipelines/template_job_$template.sh
# to nifti's
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime="00:10:00" q="d"
export jobname="nifti"
pf='_labelMA_core2D_labeled_test'
export cmd="python $scriptdir/convert/EM_stack2stack.py\
 $datadir/$dataset${pf}.h5 $datadir/$dataset${pf}.nii.gz\
 -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'\
 -y 500 -Y 1500"
source $scriptdir/pipelines/template_job_$template.sh


python $scriptdir/convert/reduceblocks.py $datadir $dataset '_maskDS' 'stack'
