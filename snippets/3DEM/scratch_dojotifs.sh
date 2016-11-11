scriptdir="${HOME}/workspace/EM"
datadir='/Users/michielk/M3_S1_GNU_NP/train'
dset_name='m000_01000-01500_01000-01500_00030-00460'

mkdir -p $datadir/tifs
mkdir -p $datadir/tifsws

python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dset_name}.h5 \
$datadir/tifs/${dset_name}.tif \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx

python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dset_name}_ws_l0.95_u1.00_s064.h5 \
$datadir/tifsws/${dset_name}.tif \
-f 'stack' -g 'stack' -s 20 20 20 -i zyx -l zyx
