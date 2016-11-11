###=========================================================================###
### environment prep
###=========================================================================###
export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
# source activate scikit-image-devel_0.13
# conda install h5py scipy
# pip install nibabel

export scriptdir="${HOME}/workspace/EM"
export datadir="${DATA}/EM/M3/M3_S1_GNU/restored" && cd $datadir
export dataset="m000"
xmax=5217; ymax=4460; zmax=460;
xs=1000; ys=1000; zs=430;
xm=0; ym=0; zm=0;
z=30; Z=460;

unset datastems
declare -a datastems
i=0
for x in `seq 0 $xs $xmax`; do
    [ $x == 5000 ] && X=$xmax || X=$((x+xs))
    for y in `seq 0 $ys $ymax`; do
        [ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
        xrange=`printf %05d ${x}`-`printf %05d ${X}`
        yrange=`printf %05d ${y}`-`printf %05d ${Y}`
        zrange=`printf %05d ${z}`-`printf %05d ${Z}`
        echo ${dataset}_${xrange}_${yrange}_${zrange}
        datastems[$i]=${dataset}_${xrange}_${yrange}_${zrange}
        i=$((i+1))
    done
done
export datastems


###=========================================================================###
### base data blocks
###=========================================================================###
# to nifti's
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

export jobname="nifti"
pf=
export cmd="python $scriptdir/convert/EM_stack2stack.py\
 $datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz\
 -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' -u"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### maskDS, maskMM, maskMM-0.02, maskMB
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

export jobname="maskDS"
export cmd="python $scriptdir/convert/prob2mask.py\
 $datadir datastem\
 -p \"\" 'stack' -l 0 -u 10000000 -o '_maskDS'"
source $scriptdir/pipelines/template_job_$template.sh
export jobname="maskMM"
export cmd="python $scriptdir/convert/prob2mask.py\
 $datadir datastem\
 -p '_probs0_eed2' 'stack' -l 0.2 -s 100000 -d 1 -o '_maskMM'"
source $scriptdir/pipelines/template_job_$template.sh
export jobname="maskMM002"
export cmd="python $scriptdir/convert/prob2mask.py\
 $datadir datastem\
 -p '_probs0_eed2' 'stack' -l 0.02 -o '_maskMM-0.02'"
source $scriptdir/pipelines/template_job_$template.sh
export jobname="maskMB"
export cmd="python $scriptdir/convert/prob2mask.py\
 $datadir datastem\
 -p '_probs' 'volume/predictions' -c 3 -l 0.3 -o '_maskMB'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### 2D connected components in maskMM
###=========================================================================###
unset datastems
declare -a datastems
datastems[0]=$dataset
export datastems

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=12000 wtime="02:00:00" q="d"
export jobname="conncomp"
export cmd="python $scriptdir/supervoxels/conn_comp.py\
 $datadir datastem\
 --maskMM '_maskMM' 'stack' -o '_labelMA_core2D' -d '2D' -i 0 -l"
source $scriptdir/pipelines/template_job_$template.sh

export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=128000 wtime="00:30:00" q=""
export jobname="conncomp"
export cmd="python $scriptdir/supervoxels/conn_comp.py\
 $datadir datastem -o '_labelMA_core2D'"
source $scriptdir/pipelines/template_job_$template.sh

# # to nifti's
# export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
# export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime="00:10:00" q="d"
#
# export jobname="nifti"
# pf="_labelMA_core2D_labeled"
# export cmd="python $scriptdir/convert/EM_stack2stack.py\
#  $datadir/datastem${pf}.h5 $datadir/datastem${pf}_tmpsubset.h5\
#  -e 0.05 0.0073 0.0073 -i 'zyx' -l 'zyx'\
#  -x 1500 -X 2500 -y 1500 -Y 2500 -d uint32"
# source $scriptdir/pipelines/template_job_$template.sh
#
# # to nifti's
# export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
# export njobs=1 nodes=1 tasks=1 memcpu=6000 wtime="00:10:00" q="d"
# export jobname="nifti"
# pf="_labelMA_core2D_labeled_tmpsubset"
# export cmd="python $scriptdir/convert/EM_stack2stack.py\
#  $datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz\
#  -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' -d uint32"
# source $scriptdir/pipelines/template_job_$template.sh

# split for proofreading
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=4000 wtime="00:10:00" q="d"
pf='_labelMA_core2D_labeled'
export jobname="split"
export cmd="python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}${pf}.h5 $datadir/datastem${pf}.h5 \
-f 'stack' -g 'stack' -s 20 20 20 -b 0 0 30 -i zyx -l zyx -p datastem"
source $scriptdir/pipelines/template_job_$template.sh
# to nifti's
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"
export jobname="nifti"
pf='_labelMA_core2D_labeled'
export cmd="python $scriptdir/convert/EM_stack2stack.py\
 $datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz\
 -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'"
source $scriptdir/pipelines/template_job_$template.sh

# pf='_labelMA_core2D_labeled'
# i=12
# itksnap -g ${datastems[i]}.nii.gz -s ${datastems[i]}${pf}.nii.gz

export deletefile="$datadir/m000_labelMA_core2D_delete.txt"
echo "m000_00000-01000_00000-01000_00030-00460: " > $deletefile
echo "m000_00000-01000_01000-02000_00030-00460: " >> $deletefile
echo "m000_00000-01000_02000-03000_00030-00460: " >> $deletefile
echo "m000_00000-01000_03000-04000_00030-00460: " >> $deletefile
echo "m000_00000-01000_04000-04460_00030-00460: " >> $deletefile
echo "m000_01000-02000_00000-01000_00030-00460: " >> $deletefile
echo "m000_01000-02000_01000-02000_00030-00460: " >> $deletefile
echo "m000_01000-02000_02000-03000_00030-00460: " >> $deletefile
echo "m000_01000-02000_03000-04000_00030-00460: " >> $deletefile
echo "m000_01000-02000_04000-04460_00030-00460: " >> $deletefile
echo "m000_02000-03000_00000-01000_00030-00460: " >> $deletefile
echo "m000_02000-03000_01000-02000_00030-00460: " >> $deletefile
echo "m000_02000-03000_02000-03000_00030-00460: " >> $deletefile
echo "m000_02000-03000_03000-04000_00030-00460: " >> $deletefile
echo "m000_02000-03000_04000-04460_00030-00460: " >> $deletefile
echo "m000_03000-04000_00000-01000_00030-00460: " >> $deletefile
echo "m000_03000-04000_01000-02000_00030-00460: " >> $deletefile
echo "m000_03000-04000_02000-03000_00030-00460: " >> $deletefile
echo "m000_03000-04000_03000-04000_00030-00460: " >> $deletefile
echo "m000_03000-04000_04000-04460_00030-00460: " >> $deletefile
echo "m000_04000-05000_00000-01000_00030-00460: " >> $deletefile
echo "m000_04000-05000_01000-02000_00030-00460: " >> $deletefile
echo "m000_04000-05000_02000-03000_00030-00460: " >> $deletefile
echo "m000_04000-05000_03000-04000_00030-00460: " >> $deletefile
echo "m000_04000-05000_04000-04460_00030-00460: " >> $deletefile
echo "m000_05000-05217_00000-01000_00030-00460: " >> $deletefile
echo "m000_05000-05217_01000-02000_00030-00460: " >> $deletefile
echo "m000_05000-05217_02000-03000_00030-00460: " >> $deletefile
echo "m000_05000-05217_03000-04000_00030-00460: " >> $deletefile
echo "m000_05000-05217_04000-04460_00030-00460: " >> $deletefile

###=========================================================================###
### connected components in maskMM-0.02
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=6 nodes=1 tasks=5 memcpu=12000 wtime="01:00:00" q=""

export jobname="conncomp"
export cmd="python $scriptdir/supervoxels/conn_comp.py\
 $datadir datastem\
 --maskMM '_maskMM-0.02' 'stack' -o '_labelMA_core'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### manual deselections from _labelMA_core (takes about an hour for m000)
###=========================================================================###
# to nifti's
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

export jobname="nifti"
pf='_labelMA_core'
export cmd="python $scriptdir/convert/EM_stack2stack.py\
 $datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz\
 -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'"
source $scriptdir/pipelines/template_job_$template.sh

export deletefile="$datadir/m000_labelMA_core_delete.txt"
echo "m000_00000-01000_00000-01000_00030-00460: 1" > $deletefile
echo "m000_00000-01000_01000-02000_00030-00460: 19 25 28" >> $deletefile
echo "m000_00000-01000_02000-03000_00030-00460: 58" >> $deletefile
echo "m000_00000-01000_03000-04000_00030-00460: 1 61" >> $deletefile
echo "m000_00000-01000_04000-04460_00030-00460: 8 12" >> $deletefile
echo "m000_01000-02000_00000-01000_00030-00460: 8 2 23 62" >> $deletefile
echo "m000_01000-02000_01000-02000_00030-00460: 26 45 43" >> $deletefile
echo "m000_01000-02000_02000-03000_00030-00460: 8 32 35 33" >> $deletefile
echo "m000_01000-02000_03000-04000_00030-00460: 1 35 54 55 81 82" >> $deletefile
echo "m000_01000-02000_04000-04460_00030-00460: 2 24" >> $deletefile
echo "m000_02000-03000_00000-01000_00030-00460: 9 30 55 57" >> $deletefile
echo "m000_02000-03000_01000-02000_00030-00460: 14 38 45 55" >> $deletefile
echo "m000_02000-03000_02000-03000_00030-00460: 12 29 40 69 68 74" >> $deletefile
echo "m000_02000-03000_03000-04000_00030-00460: 17 25 39 35 45 56 55 67 77" >> $deletefile
echo "m000_02000-03000_04000-04460_00030-00460: 4 1 12 25 26 27" >> $deletefile
echo "m000_03000-04000_00000-01000_00030-00460: 28 41" >> $deletefile
echo "m000_03000-04000_01000-02000_00030-00460: 31" >> $deletefile
echo "m000_03000-04000_02000-03000_00030-00460: 1 28 52" >> $deletefile
echo "m000_03000-04000_03000-04000_00030-00460: 36 63 66" >> $deletefile
echo "m000_03000-04000_04000-04460_00030-00460: 1 18 30 32 36 39 44" >> $deletefile
echo "m000_04000-05000_00000-01000_00030-00460: 21" >> $deletefile
echo "m000_04000-05000_01000-02000_00030-00460: 48" >> $deletefile
echo "m000_04000-05000_02000-03000_00030-00460: 35 38 46 49" >> $deletefile
echo "m000_04000-05000_03000-04000_00030-00460: 34 13 46" >> $deletefile
echo "m000_04000-05000_04000-04460_00030-00460: 8 14 18 21 30 24 33 36 35" >> $deletefile
echo "m000_05000-05217_00000-01000_00030-00460: 1 3" >> $deletefile
echo "m000_05000-05217_01000-02000_00030-00460: 1 4 5" >> $deletefile
echo "m000_05000-05217_02000-03000_00030-00460: 1 2 4 5" >> $deletefile
echo "m000_05000-05217_03000-04000_00030-00460: 2 3 5 7" >> $deletefile
echo "m000_05000-05217_04000-04460_00030-00460: 1 2" >> $deletefile

export template='array' additions='dellabels'
export njobs=2 nodes=1 tasks=15 memcpu=3000 wtime="00:10:00" q="d"

export jobname="dellabels"
export cmd="python $scriptdir/supervoxels/delete_labels.py\
 $datadir datastem\
 -l '_labelMA_core' 'stack' -d deletelabels -o '_manedit'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### watershed on prob_ics
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=6 nodes=1 tasks=5 memcpu=25000 wtime="02:00:00" q=""

# l=0.95; u=1.00; s=064;
# svoxpf="_ws_l${l}_u${u}_${s}"
# export jobname="ws${svoxpf}"
# export cmd="python $scriptdir/supervoxels/EM_watershed.py\
#  $datadir datastem\
#  --maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack'\
#  -p '_probs' 'volume/predictions'\
#  -c 1 -l $l -u $u -s $s -o ${svoxpf}"
# source $scriptdir/pipelines/template_job_$template.sh
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_${s}"
export jobname="ws${svoxpf}"
export cmd="python $scriptdir/supervoxels/EM_watershed.py\
 $datadir datastem\
 --maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack'\
 -p '_probs' 'volume/predictions' -c 1\
 -l $l -u $u -s $s -o ${svoxpf}"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### agglomerate watershedMA
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

l=0.95; u=1.00; s=064;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="agglo${svoxpf}"
export cmd="python $scriptdir/supervoxels/agglo_from_labelmask.py\
 ${datadir} datastem\
 -l '_labelMA_core_manedit' 'stack' -s ${svoxpf} 'stack'\
 -o '_labelMA' -m '_maskMA'"
source $scriptdir/pipelines/template_job_$template.sh
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="agglo${svoxpf}"
export cmd="python $scriptdir/supervoxels/agglo_from_labelmask.py\
 $datadir datastem\
 -l '_labelMA_core_manedit' 'stack' -s ${svoxpf} 'stack'\
 -o '_labelMA' -m '_maskMA'"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### manual deselections from _labelMA (takes about an hour for m000)
###=========================================================================###
# to nifti's
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

export jobname="nifti"
pf="_labelMA${svoxpf}"
export cmd="python $scriptdir/convert/EM_stack2stack.py\
 $datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz\
 -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'"
source $scriptdir/pipelines/template_job_$template.sh

export deletefile="$datadir/m000_labelMA_agglo_delete.txt"
echo "m000_00000-01000_00000-01000_00030-00460: 847 525 57 4976 8024 16362 9051 15223 20321 22537" > $deletefile
echo "m000_00000-01000_01000-02000_00030-00460: 13016 25029 28824 15625" >> $deletefile
echo "m000_00000-01000_02000-03000_00030-00460: 1735 10849 4405 26369 24212" >> $deletefile
echo "m000_00000-01000_03000-04000_00030-00460: 574 100 45 2865 2570 5389 3693 11654 6075 19329 25829" >> $deletefile
echo "m000_00000-01000_04000-04460_00030-00460: 54 289 1376" >> $deletefile
echo "m000_01000-02000_00000-01000_00030-00460: 778 5738 16587 29044 31073 22297" >> $deletefile
echo "m000_01000-02000_01000-02000_00030-00460: 1799 123 55503" >> $deletefile
echo "m000_01000-02000_02000-03000_00030-00460: 5306 76 144 143 40439 43599 49353 62579" >> $deletefile
echo "m000_01000-02000_03000-04000_00030-00460: 387 227 181 68 374 26543 15755 42641 47065 34330 16 49280" >> $deletefile
echo "m000_01000-02000_04000-04460_00030-00460: 127 2017 4695" >> $deletefile
echo "m000_02000-03000_00000-01000_00030-00460: 293 4064 8859 37118 39285" >> $deletefile
echo "m000_02000-03000_01000-02000_00030-00460: 211 157 21160 39459 357 30055 44642 31477 164 48826 310 31477" >> $deletefile
echo "m000_02000-03000_02000-03000_00030-00460: 152 1704 4794 6065 20699 27693 26960 34358 46269 339 32492 20747" >> $deletefile
echo "m000_02000-03000_03000-04000_00030-00460: 510 7159 13441 5779 24223 32034 41102 31695 41463 44441 42112" >> $deletefile
echo "m000_02000-03000_04000-04460_00030-00460: 11572 14390 15533" >> $deletefile
echo "m000_03000-04000_00000-01000_00030-00460: 20 51482 46615" >> $deletefile
echo "m000_03000-04000_01000-02000_00030-00460: 1322 408 31756 408 53847 39757" >> $deletefile
echo "m000_03000-04000_02000-03000_00030-00460: 51091 23540" >> $deletefile
echo "m000_03000-04000_03000-04000_00030-00460: 390 379" >> $deletefile
echo "m000_03000-04000_04000-04460_00030-00460: 3052 3206 6363 5912 4 7815 101 5911" >> $deletefile
echo "m000_04000-05000_00000-01000_00030-00460: 30 201 136" >> $deletefile
echo "m000_04000-05000_01000-02000_00030-00460: 391 9958 149 20866 498 22602" >> $deletefile
echo "m000_04000-05000_02000-03000_00030-00460: 200 12 13462 24062 28768" >> $deletefile
echo "m000_04000-05000_03000-04000_00030-00460: 98 12 1 14846 212 13346 22184 23051" >> $deletefile
echo "m000_04000-05000_04000-04460_00030-00460: 17 6097 6494" >> $deletefile
echo "m000_05000-05217_00000-01000_00030-00460: " >> $deletefile
echo "m000_05000-05217_01000-02000_00030-00460: " >> $deletefile
echo "m000_05000-05217_02000-03000_00030-00460: " >> $deletefile
echo "m000_05000-05217_03000-04000_00030-00460: 1" >> $deletefile
echo "m000_05000-05217_04000-04460_00030-00460: " >> $deletefile
export mergefile="$datadir/m000_labelMA_agglo_merge.txt"
echo "m000_00000-01000_01000-02000_00030-00460: 595 1883" > $mergefile
echo "m000_00000-01000_03000-04000_00030-00460: 6075 17575" >> $mergefile
echo "m000_00000-01000_04000-04460_00030-00460: 3621 1377" >> $mergefile
echo "m000_02000-03000_02000-03000_00030-00460: 13629 26960" >> $mergefile
echo "m000_02000-03000_03000-04000_00030-00460: 20173 365" >> $mergefile
echo "m000_03000-04000_03000-04000_00030-00460: 33587 50572" >> $mergefile
echo "m000_04000-05000_01000-02000_00030-00460: 285 27565" >> $mergefile

export template='array' additions='deletelabels-mergelabels'
export njobs=2 nodes=1 tasks=15 memcpu=3000 wtime="00:10:00" q="d"

l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="dellabels"
export cmd="python $scriptdir/supervoxels/delete_labels.py\
 $datadir datastem\
 -l '_labelMA${svoxpf}' 'stack' -d deletelabels -m mergelabels -o '_manedit'"
source $scriptdir/pipelines/template_job_$template.sh

###=========================================================================###
### fill holes in myelinated axons (NB: hardly does anything!)
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="10:00:00" q=""

# l=0.95; u=1.00; s=064;
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="fill${svoxpf}"
export cmd="python $scriptdir/supervoxels/fill_holes.py\
 $datadir datastem\
 -l '_labelMA${svoxpf}' 'stack' -m '_maskMA' 'stack'\
 --maskMM '_maskMM' 'stack' --maskMA '_maskMA' 'stack'\
 -o '_filled' -p '_holes' -w 2"
source $scriptdir/pipelines/template_job_$template.sh

# to nifti's
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

export jobname="nifti"
pf="_labelMA${svoxpf}_filled"
export cmd="python $scriptdir/convert/EM_stack2stack.py\
 $datadir/datastem${pf}.h5 $datadir/datastem${pf}.nii.gz\
 -e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### separate sheaths
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="10:00:00" q=""

# l=0.95; u=1.00; s=064;
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="sheaths"
export cmd="python $scriptdir/mesh/EM_separate_sheaths.py\
 $datadir datastem\
 -l '_labelMA${svoxpf}_manedit' 'stack'\
 --maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack'\
 -w -d"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### Neuroproof agglomeration
###=========================================================================###
export template='array' additions='neuroproof' CONDA_ENV="neuroproof-test"
export njobs=30 nodes=1 tasks=1 memcpu=60000 wtime="01:00:00" q=""

NPdir=${HOME}/workspace/Neuroproof_minimal
trainset="m000_01000-01500_01000-01500_00030-00460"
svoxpf='_ws_l0.99_u1.00_s005_labelMA_filled'
classifier="_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallel"; cltype='h5';
thr=0.1; alg=1;
# cp ${DATA}/EM/Neuroproof/M3_S1_GNU_NP/train/${trainset}${classifier}.${cltype} .
export jobname="NP"
export cmd="$NPdir/NeuroProof_stack\
 -watershed $datadir/datastem${svoxpf}.h5 'stack'\
 -prediction $datadir/datastem_probs.h5 'volume/predictions'\
 -output $datadir/datastem_prediction${classifier}${cltype}_thr${thr}_alg${alg}M.h5 'stack'\
 -classifier $datadir/${trainset}${classifier}.${cltype}\
 -threshold ${thr} -algorithm ${alg}"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### classify neurons MA/UA
###=========================================================================###
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8

export template='mpi'
export njobs=30 nodes=4 tasks=16 memcpu=6000 wtime="10:00:00" q=""

export jobname="classMA-UA"
export cmd="python $scriptdir/mesh/EM_classify_neurons.py\
 $datadir datastem\
 --supervoxels '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.1_alg1' 'stack'\
 -o '_per' -m"
source $scriptdir/pipelines/template_job_$template.sh






###=========================================================================###
### EED probs3_eed2
###=========================================================================###
module load hdf5-parallel/1.8.14_mvapich2_intel
module load python/2.7__gcc-4.8
module load matlab/R2015a

mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b

export template='array' additions=""
export njobs=15 nodes=1 tasks=2 memcpu=50000 wtime="05:00:00" q=""

export jobname="EED"
layer=4
export cmd="$datadir/bin/EM_eed\
 '$datadir' 'datastem_probs' '/volume/predictions' '/stack' $layer\
 > $datadir/datastem_probs.log"
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### merge blocks
###=========================================================================###
export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime="01:30:00" q="d"

pf=_maskDS; field='stack'
pf=_maskMM; field='stack'
export jobname="merge"
export cmd="python $scriptdir/convert/EM_mergeblocks.py\
 $datadir $datadir/${dataset}${pf}.h5\
 -i ${datastems[*]} -t ${pf}\
 -f $field -l 'zyx' -b 0 0 30\
 -p $xs $ys $zs -q $xm $ym $zm -s $xmax $ymax $zmax"
source $scriptdir/pipelines/template_job_$template.sh
