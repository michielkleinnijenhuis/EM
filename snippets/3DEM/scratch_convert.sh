###=========================================================================###
### cutouts and conversions ###
###=========================================================================###
scriptdir="$HOME/workspace/EM"
datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP'

dataset=m000
oX=1000;oY=1000;oZ=30;
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
nx=1000; nX=1500; ny=1000; nY=1500; nz=30; nZ=460;
# nx=1000; nX=1250; ny=1000; nY=1250; nz=200; nZ=300;
# nx=1000; nX=1100; ny=1000; nY=1100; nz=200; nZ=250;
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`

python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/orig/${datastem}.h5 \
$datadir/train/${newstem}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))

pf='_PA'
pf='_probs0_eed2'
pf='_ws_l0.95_u1.00_s064'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/orig/${datastem}${pf}.h5 \
$datadir/train/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))

python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/orig/${datastem}_slic_s00500_c2.000_o0.050.h5 \
$datadir/train/${newstem}_slic_s00500_c2.000_o0.050.h5 \
-e 0.05 0.0073 0.0073 \
-i 'zyx' -l 'zyx' -d 'int32' \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))
mv $datadir/train/${newstem}_slic_s00500_c2.000_o0.h5 $datadir/train/${newstem}_slic_s00500_c2.000_o0.050.h5

python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/orig/${datastem}_probs.h5 \
$datadir/train/${newstem}_probs.h5 \
-f 'volume/predictions' -g 'volume/predictions' \
-e 0.05 0.0073 0.0073 1 \
-i 'zyxc' -l 'zyxc' \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))


for thr in 0.2 0.3 0.4 0.5; do
python $scriptdir/convert/EM_stack2stack.py \
$datadir/train/${newstem}_prediction_thr${thr}_alg1.h5 \
$datadir/train/${newstem}_prediction_thr${thr}_alg1.nii.gz \
-e 0.05 0.0073 0.0073 \
-i 'zyx' -l 'zyx'
done


datastem=train/m000_01000-01500_01000-01500_00030-00460
datastem=test/m000_03000-04000_03000-04000_00030-00460
scriptdir="$HOME/workspace/EM"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}.h5 \
$datadir/${datastem}.nii.gz \
-i 'zyx' -l 'zyx' -e 0.05 -0.0073 -0.0073 -u





## local
scriptdir="$HOME/workspace/EM"

datadir='/Users/michielk/M3_S1_GNU_NP/train'
datastem='m000_01000-01500_01000-01500_00030-00460'
datastem='m000_01000-01250_01000-01250_00200-00300'

datadir='/Users/michielk/M3_S1_GNU_NP/test'
datastem='m000_02000-03000_02000-03000_00030-00460'
datastem='m000_03000-04000_03000-04000_00030-00460'

python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}.h5 \
$datadir/${datastem}.nii.gz \
-i 'zyx' -l 'xyz' -e 0.0073 0.0073 0.05 -u

pf="_ws_l0.95_u1.00_s064"
pf="_ws_l0.99_u1.00_s005"
pf="_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1"
# pf="_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.3_alg1M"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'
