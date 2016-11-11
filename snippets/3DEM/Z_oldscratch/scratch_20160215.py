scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
paraview --state=$datadir/m000_01000-01500_01000-01500_00200-00300/stack+mesh_compact.pvsm


qsubfile=$datadir/EM_con2.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_con" >> $qsubfile
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000.h5 ${datadir}/m000.h5 \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u"  >> $qsubfile
sbatch $qsubfile

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
pf=''
xs=1000; ys=1000;
z=30; Z=460;
for x in 2000 3000; do
for y in 2000 3000; do
X=$((x+xs))
Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
gunzip ${datadir}/${datastem}${pf}.nii.gz
done
done
