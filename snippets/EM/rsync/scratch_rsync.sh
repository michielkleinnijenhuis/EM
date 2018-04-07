remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test"
scriptdir="${HOME}/workspace/EM"
datadir=$localdir
xe=0.0511; ye=0.0511; ze=0.05;

cd $localdir

for pf in '' '_labelMA_core2D' '_maskMB' '_maskMM-0.02'; do
f="M3S1GNU_01950-03050_02950-04050_00030-00460${pf}.h5"
rsync -Pavz $remdir/${f} $localdir
done

for f in M3S1GNUds7_labelMA_core2D.h5 \
M3S1GNUds7_labelMA_core2D_fw_nf_label.h5 \
M3S1GNUds7_labelMA_core2D_fw_nf_euler_number.h5 \
M3S1GNUds7_labelMA_core2D_fw_nf_area.h5 \
M3S1GNUds7_labelMA_core2D_fw_nf_extent.h5 \
M3S1GNUds7_labelMA_core2D_fw_nf_mean_intensity.h5 \
M3S1GNUds7_labelMA_core2D_fw_nf_solidity.h5 \
M3S1GNUds7_labelMA_core2D_fw_3Dlabeled.h5; do
# rsync -avz $remdir/${f} $localdir
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${f}" \
$datadir/"${f/.h5/.nii}" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done

f=M3S1GNUds7_labelMA_core2D_fw_3Diter3_closed.h5



remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/train"
localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/viz"
scriptdir="${HOME}/workspace/EM"
datadir=$localdir

f=m000_01000-01500_01000-01500_00030-00460_ws_l0.99_u1.00_s010.h5
rsync -avz $remdir/${f} $localdir
