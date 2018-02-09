###=========================================================================###
### label2stl
###=========================================================================###
### on local machine
scriptdir="$HOME/workspace/EM"
datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
datastem='M3S1GNUds7'

python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L '_labelMF_final' -f 'stack' -o 30 0 0
python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L '_labelMA_final' -f 'stack' -o 30 0 0

### on arcus-b
host="ndcn0180@arcus-b.arc.ox.ac.uk"
remdir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old/ds7_arc"
localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
rsync -avz "$host:$localdir/*_final.h5" $remdir

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

# rsync -avz "$host:$remdir/M3S1GNUds7/dmcsurf_1-1-1/*.stl" "$localdir/M3S1GNUds7/dmcsurf_1-1-1/"


###=========================================================================###
### stl2blender
###=========================================================================###
### on local machine
comp='_labelGL_final'
blender -b -P $scriptdir/mesh/stl2blender.py -- \
"$datadir/$datastem/dmcsurf_1-1-1" ${comp} -L ${comp} \
-s 100 0.1 True True True -d 0.02 -e 0

### on local machine using multiple cores (needs merging blend-files)
# for comp in '_labelMF_final' '_labelMA_final' '_labelGL_final'; do
# for d in 0.20 0.10 0.05 0.02; do
comp='_labelGL_final'
d=0.02
stldir="$datadir/$datastem/dmcsurf_1-1-1"
stlfiles=($(find $stldir/ -maxdepth 1 -name "${comp}*.stl"))
nfiles=${#stlfiles[@]}
np=6
nblock=$((nfiles/np))
os=0
echo $nfiles
for p in `seq 1 $np`; do
if [[ "$p" == 6 ]]
then stlblock=("${stlfiles[@]:$os}")
else  stlblock=("${stlfiles[@]:$os:$nblock}")
fi
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$stldir ${comp}_d${d}collapse_s100-0.1_$p -L ${comp} \
-S ${stlblock[@]} \
-s 100 0.1 True True True -d $d -e 0 &
os=$((os+nblock))
stlblock=''
done
# done
# done
# -l 100 0.05 0.01 True True True False True -d 0.05 -e 0.01 &


### on arcus (needs merging blend-files)
export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
module load blender/2.67b

datadir="${DATA}/EM/M3/M3_S1_GNU_old/ds7_arc" && cd $datadir
datastem='M3S1GNUds7'

comp='_labelUA_final'
d=0.02
stldir="$datadir/$datastem/dmcsurf_1-1-1"
stlfiles=($(find $stldir/ -maxdepth 1 -name "${comp}*.stl"))
nfiles=${#stlfiles[@]}
np=32
nblock=$((nfiles/np))
os=0
for p in `seq 1 $np`; do
if [[ "$p" == "$np" ]]
then stlblock=("${stlfiles[@]:$os}")
else  stlblock=("${stlfiles[@]:$os:$nblock}")
fi
echo "blender -b -P $scriptdir/mesh/stl2blender.py -- \
$stldir ${comp}_d${d}collapse_s100-0.1_$p -L ${comp} \
-S ${stlblock[@]} -s 100 0.1 True True True -d $d -e 0" > script_$p.sh
os=$((os+nblock))
stlblock=''
done

qsubfile=script.sh
echo '#!/bin/bash' > $qsubfile
echo '#PBS -l nodes=2:ppn=16' >> $qsubfile
echo '#PBS -l walltime=10:00:00' >> $qsubfile
echo '#PBS -N em_stl2blend' >> $qsubfile
echo '#PBS -V' >> $qsubfile
echo 'cd $PBS_O_WORKDIR' >> $qsubfile
echo "$datadir/script_\$PBS_ARRAYID.sh" >> $qsubfile

qsub -t 1-32 script.sh

qsub -q develq -t 1-3 $qsubfile


## UA blends
scriptdir="$HOME/workspace/EM"
datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
datastem='M3S1GNUds7'

comp='_labelUA_final'
blender -b -P $scriptdir/mesh/stl2blender.py -- \
"$datadir/$datastem/dmcsurf_1-1-1" ${comp} -L ${comp} \
-s 100 0.1 True True True -d 0.02 -e 0
