TODO:
- myelin controlpoints
- segment supervoxels (manual and automatic)
- enforce connected axon segments
-


#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_slic_s500_c10_o0.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
pf='_slic_s500_c10_o0'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000_01000-02000_01000-02000_00030-00460${pf}.h5 ${datadir}/m000_01000-02000_01000-02000_00030-00460${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05


python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000_01000-01500_01000-01500_00200-00300.h5 ${datadir}/m000_01000-01500_01000-01500_00200-00300.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
pf='_slic_s500_c0.01_o0'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000_01000-01500_01000-01500_00200-00300${pf}.h5 ${datadir}/m000_01000-01500_01000-01500_00200-00300${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32


python $HOME/workspace/EM/convert/EM_stack2stack.py \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0.h5 \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0_tmp.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
pf='_slicvoxels002'
python $HOME/workspace/EM/convert/EM_stack2stack.py \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0${pf}.h5 \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0_tmp${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32



scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"

oX=1000;oY=1000;oZ=30;
nx=1000;nX=1500;ny=1000;nY=1500;nz=30;nZ=80;
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
# raw data to uint8
pf=''
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
# other
pf='_slic_s00500_c0.020_o0.010'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 -0.0073 -0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32
