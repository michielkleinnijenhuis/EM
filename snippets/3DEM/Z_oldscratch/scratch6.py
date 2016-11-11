scriptdir="$HOME/workspace/EM"
# DATA="$HOME/oxdata/P01"
datadir="$DATA/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
refsect='0250'

pf=; field='stack'
pf=_probs; field='volume/predictions'
pf=_probs0_eed2; field='stack'
qsubfile=$datadir/EM_mb${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=02:00:00" >> $qsubfile
echo "#SBATCH --mem=250000" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py -i \
$datadir/${dataset}_00000-01000_00000-01000_00000-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_00000-01000_00000-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_00000-01000_00000-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_01000-02000_00000-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_01000-02000_00000-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_01000-02000_00000-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_02000-03000_00000-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_02000-03000_00000-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_02000-03000_00000-00460${pf}.h5 \
-o $datadir/${dataset}_00000-03000_00000-03000_00000-00430${pf}.h5 \
-f $field -l 'zyx'" >> $qsubfile
sbatch -p compute $qsubfile

#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_00000-03000_?????-?????_?????-?????.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
pf=;
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_00000-03000_00000-03000_00000-00430.h5" \
"${datadir}/${dataset}_00000-03000_00000-03000_00000-00430.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u


x=0; X=3000; y=0; Y=3000; z=0; Z=430; layer=1;
qsubfile=$datadir/EM_eed_submit_${x}-${X}_${y}-${Y}_${layer}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
echo "$datadir/bin/EM_eed '$datadir' \
'${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs' \
'/volume/predictions' '/stack' $layer \
> $datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs.log &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile


x=0000; X=3000; y=0000; Y=3000; z=0; Z=430;  # mem +- 188GB for MA
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=24:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' \
-n 5 -o 220 235 491 -s 430 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
sbatch -p devel $qsubfile

--SEfile '_seg.h5' --MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5'
















scriptdir="$HOME/workspace/EM"
# DATA="$HOME/oxdata/P01"
datadir="$DATA/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
refsect='0250'

rename _00000-00460.h5 _00030-00460.h5 ${dataset}_?????-?????_?????-?????_00000-00460.h5
rename m000_05000-06000 m000_05000-05217 ${dataset}_05000-06000_?????-?????_?????-?????.h5
rename _04000-05000_00030-00460.h5 _04000-04460_00030-00460.h5 ${dataset}_?????-?????_04000-05000_?????-?????.h5

rename _00000-00460 _00030-00460 ${dataset}_?????-?????_?????-?????_00000-00460_probs.*
rename m000_05000-06000 m000_05000-05217 ${dataset}_05000-06000_?????-?????_?????-?????_probs.*
rename _04000-05000_00030-00460 _04000-04460_00030-00460 ${dataset}_?????-?????_04000-05000_?????-?????_probs.*

rename _00000-00460 _00030-00460 ${dataset}_?????-?????_?????-?????_00000-00460_probs0_eed2.*
rename m000_05000-06000 m000_05000-05217 ${dataset}_05000-06000_?????-?????_?????-?????_probs0_eed2.*
rename _04000-05000_00030-00460 _04000-04460_00030-00460 ${dataset}_?????-?????_04000-05000_?????-?????_probs0_eed2.*

rename _00000-00460 _00030-00460 m000_*
rename m000_05000-06000 m000_05000-05217 m000_*
rename _04000-05000_00030-00460 _04000-04460_00030-00460 m000_*

pf=; field='stack'
pf=_probs; field='volume/predictions'
pf=_probs0_eed2; field='stack'
qsubfile=$datadir/EM_mb${pf}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=02:00:00" >> $qsubfile
#echo "#SBATCH --mem=50000" >> $qsubfile
echo "#SBATCH --job-name=EM_mb" >> $qsubfile
echo "python $scriptdir/convert/EM_mergeblocks.py -i \
$datadir/${dataset}_00000-01000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_00000-01000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_01000-02000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_02000-03000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_03000-04000_00030-00460${pf}.h5 \
$datadir/${dataset}_00000-01000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_01000-02000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_02000-03000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_03000-04000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_04000-05000_04000-04460_00030-00460${pf}.h5 \
$datadir/${dataset}_05000-05217_04000-04460_00030-00460${pf}.h5 \
-o $datadir/${dataset}_00000-05217_00000-04460_00030-00460${pf}.h5 \
-f $field -l 'zyx'" >> $qsubfile
sbatch -p compute $qsubfile
