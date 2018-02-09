### EED regionrefs
scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
datadir="$HOME/oxdata/P01/EM/scratch_wmem_package/ds7_arc"

import os
import h5py

xmax=1311; ymax=1255; zmax=430;
xs=438; ys=419; zs=430;
xm=0; ym=0; zm=0;

datadir="/Users/michielk/oxdata/P01/EM/scratch_wmem_package/ds7_arc"
h5_path = os.path.join(datadir, 'M3S1GNUds7.h5')
f = h5py.File(h5_path, 'a')
ds = f['stack']
roisets = f.require_group('ROIsets')
roiname = 'blocks'
refs = create_regrefs(roisets, roiname, slices)
f.close()


### MERGEBLOCKS
ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/archive/M3_S1_GNU_old/blocks"
localdir="$HOME/oxdata/P01/EM/scratch_wmem_package"
for f in M3S1GNU_00000/M3S1GNU_00000-01050_00000-01050_00030-00460_probs0_eed2.h5 \
M3S1GNU_00000/M3S1GNU_00000-01050_00950-02050_00030-00460_probs0_eed2.h5 \
M3S1GNU_00950/M3S1GNU_00950-02050_00000-01050_00030-00460_probs0_eed2.h5 \
M3S1GNU_00950/M3S1GNU_00950-02050_00950-02050_00030-00460_probs0_eed2.h5; do
rsync -Pavz $remdir/${f} $localdir
# rsync -avz $localdir/${f} $remdir
done

scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
datadir="$HOME/oxdata/P01/EM/scratch_wmem_package"
xmax=2000; ymax=2000; zmax=460;
xs=1000; ys=1000; zs=430;
xm=50; ym=50; zm=0;
z=30; Z=460;
xo=0; yo=0; zo=30;
xe=0.0073; ye=0.0073; ze=0.05;

python $scriptdir/wmem/mergeblocks.py \
$datadir/M3S1GNU_00000-01050_00000-01050_00030-00460_probs0_eed2.h5/stack \
$datadir/M3S1GNU_00000-01050_00950-02050_00030-00460_probs0_eed2.h5/stack \
$datadir/M3S1GNU_00950-02050_00000-01050_00030-00460_probs0_eed2.h5/stack \
$datadir/M3S1GNU_00950-02050_00950-02050_00030-00460_probs0_eed2.h5/stack \
$datadir/M3S1GNU_00000-02000_00000-02000_00030-00460_probs0_eed2.h5/stack \
-b 0 $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax

python $scriptdir/wmem/stack2stack.py \
$datadir/M3S1GNU_00000-02000_00000-02000_00030-00460_probs0_eed2.h5/stack \
$datadir/M3S1GNU_00000-02000_00000-02000_00030-00460_probs0_eed2.nii.gz \
-z 350 -Z 400

# python $scriptdir/wmem/stack2stack.py $basepath.h5/stack $basepath.h5/cutout -x 250 -X 750 -y 250 -Y 900 -a '.tif'


import os
from wmem import mergeblocks




export template='single' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=1 nodes=1 tasks=1 memcpu=25000 wtime="00:30:00" q=""

for pf in '_maskDS' '_maskMM' '_maskMM-0.02' '_maskMB'; do
    export jobname="merge${pf}"
    export cmd="

    "
    source $scriptdir/pipelines/template_job_$template.sh
done
