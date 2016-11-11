scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"

dataset='m000'
z=30; Z=460;
x=5000
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
datafile=${datadir}/${datastem}.h5
python $scriptdir/convert/EM_stack2stack.py \
${datafile} ${datafile/.h5/.nii.gz} \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
done

for f in `ls m000_?????-?????_?????-?????_00000-00460*`; do
mv $f ${f/00000-00460/00030-00460}
done

dataset='m000'
z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
datafile=${datadir}/${datastem}.h5
python $scriptdir/convert/EM_stack2stack.py \
${datafile} ${datafile/.h5/.nii.gz} \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
done
done

dataset='m000'
z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
for pf in _probs_ws_MA_probs_ws_MAfilled; do #_seeds_MA2 _probs_ws_MA_probs_ws_MA
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done
done
done

dataset='m000'
z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
pf='_probs_ws_MM'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
