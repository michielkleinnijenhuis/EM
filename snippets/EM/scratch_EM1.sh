
mergeblocks h '_probs_eed' 'probs_eed' '_probs_eed' 'probs_eed' 41 $((289 - 41))
mergeblocks h '_probs_eed' 'sum16_eed' '_probs_sum16_eed' 'sum16_eed' 247 $((289 - 247))
mergeblocks h '_probs_eed' 'sum16_eed' '_probs_sum16_eed' 'sum16_eed' 204 $((289 - 204))


PATH=:$PATH
source activate root
export PYTHONPATH=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/pyDM3reader

import h5py
from wmem import utils

h5path_in='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00_probs_sum0247_eed.h5/sum0247_eed'
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

h5path_out='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00_probs_sum0247_eed_tmp.h5/sum0247_eed'

h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                    h5path_out,
                                    chunks=ds_in.chunks or None,
                                    element_size_um=elsize,
                                    axislabels=axlab)


ds_out[1:3, 1:3, 1:3] = ds_in[1:3, 1:3, 1:3]

h5file_in.close()
h5file_out.close()


# OK
import h5py
h5path_in='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00_probs_sum0247_eed_tmp.h5/sum0247_eed'
h5path_file='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00_probs_sum0247_eed_tmp.h5'
h5path_dset='sum0247_eed'
h5file = h5py.File(h5path_file, 'a')

# h5path_dset in h5file
h5ds = h5file[h5path_dset]
h5file.close()






pyfile=$datadir/EM_corr_script.py
echo "import os" > $pyfile
echo "import numpy as np" >> $pyfile
echo "from wmem import utils" >> $pyfile
echo "datadir = '$datadir'" >> $pyfile
echo "dataset = '$dataset'" >> $pyfile
echo "h5dset_in = dataset + '.h5/stack'" >> $pyfile
echo "h5path_in = os.path.join(datadir, h5dset_in)" >> $pyfile
echo "h5_in, ds_in, es, al = utils.h5_load(h5path_in)" >> $pyfile
echo "es = np.append(ds_in.attrs['element_size_um'], 1)" >> $pyfile
echo "h5_in.close()" >> $pyfile
for datastem in ${datastems[@]}; do
echo "h5dset_out = '$datastem' + '_probs.h5/volume/predictions'" >> $pyfile
echo "h5path_out = os.path.join(datadir, 'blocks_0500', h5dset_out)" >> $pyfile
echo "h5_out, ds_out, _, _ = utils.h5_load(h5path_out)" >> $pyfile
echo "utils.h5_write_attributes(ds_out, element_size_um=es)" >> $pyfile
echo "h5_out.close()" >> $pyfile
done

source activate root
export PYTHONPATH=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/pyDM3reader
python $pyfile


pyfile=$datadir/EM_corr_script.py
echo "import os" > $pyfile
echo "import numpy as np" >> $pyfile
echo "from wmem import utils" >> $pyfile
echo "datadir = '$datadir'" >> $pyfile
echo "dataset = '$dataset'" >> $pyfile
echo "h5dset_in = dataset + '.h5/stack'" >> $pyfile
echo "h5path_in = os.path.join(datadir, h5dset_in)" >> $pyfile
echo "h5_in, ds_in, es, al = utils.h5_load(h5path_in)" >> $pyfile
echo "es = ds_in.attrs['element_size_um']" >> $pyfile
echo "h5_in.close()" >> $pyfile
for datastem in ${datastems[@]}; do
echo "h5dset_out = '$datastem' + '_probs.h5/sum0247'" >> $pyfile
echo "h5path_out = os.path.join(datadir, 'blocks_0500', h5dset_out)" >> $pyfile
echo "h5_out, ds_out, _, _ = utils.h5_load(h5path_out)" >> $pyfile
echo "utils.h5_write_attributes(ds_out, element_size_um=es)" >> $pyfile
echo "h5_out.close()" >> $pyfile
done

source activate root
export PYTHONPATH=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/pyDM3reader
python $pyfile



import h5py
h5path_file = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU/M3S1GNU.h5'
h5file = h5py.File(h5path_file, 'a')
h5file["data"] = h5file["stack"]
del h5file["stack"]
h5file.close()




(root) [ndcn0180@login12(arcus-b) M3S1GNU]$ h5ls -v M3S1GNUds7.h5
Opened "M3S1GNUds7.h5" with sec2 driver.
data                     Dataset {460/460, 1256/1256, 1312/1312}
    Attribute: DIMENSION_LABELS {3}
        Type:      variable-length null-terminated ASCII string
        Data:  "z", "y", "x"
    Attribute: element_size_um {3}
        Type:      native double
        Data:  0.05, 0.0511, 0.0511
    Location:  1:800
    Links:     1
    Chunks:    {15, 79, 82} 194340 bytes
    Storage:   1516042240 logical bytes, 1010991400 allocated bytes, 149.96% utilization
    Filter-0:  deflate-1 OPT {4}
    Type:      native unsigned short

h5repack -i M3S1GNUds7.h5 -o M3S1GNUds7_testrepack.h5 -v -f GZIP=9 -l CHUNK=10x10x10

h5repack -i B-NT-S10-2f_ROI_00_probs_eed_sum16.h5 -o B-NT-S10-2f_ROI_00_probs_eed_sum16_repack.h5 -v -f NONE



# getting a good maskMM, testing on block
export PYTHONPATH=$scriptdir
datadir=$blockdir
dataset=B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184
dataset_ds=$dataset

h52nii '' $dataset '' 'data' '-u -i zyx -o xyz'
# h52nii '' $dataset '_probs' 'volume/predictions' '_probs' 'volume-predictions' "-u -i zyxc -o xyzc"
for i in `seq 0 7`; do
h52nii '' "$dataset" '_probs' 'volume/predictions' '_probs' "volume-predictions-$i" "-i zyxc -o xyzc -D 0 0 1 0 0 1 0 0 1 $i $((i+1)) 1 -d float"
done
h52nii '' $dataset '_probs_eed' 'sum0247_eed' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset '_probs_eed' 'sum16_eed' '' '' '-i zyx -o xyz -d float'
for i in `seq 0 7`; do
h52nii '' "$dataset" '_probs_eed' 'probs_eed' '_probs_eed' "probs_eed_$i" "-i zyxc -o xyzc -D 0 0 1 0 0 1 0 0 1 $i $((i+1)) 1 -d float"
done

# prob2mask h 1 1
ipf='_probs_eed'; ids='sum0247_eed';
opf='_masks'; ods='maskMM';
arg='-g -l 0.5 -S';
z=0; Z=184;
vol_slice=;
cmd=$( get_cmd_prob2mask )
single_job "$cmd"

h52nii '' $dataset '_masks' 'maskMM' '' '' '-i zyx -o xyz'
h52nii '' $dataset '_masks' 'maskICS' '' '' '-i zyx -o xyz'
h52nii '' $dataset '_masks' 'maskMA' '' '' '-i zyx -o xyz'
h52nii '' $dataset '_masks' 'maskDS' '' '' '-i zyx -o xyz'

ipf='_masks'; ids='maskMM';
opf='_masks'; ods='maskMM_testfilter';
arg='-g -l 0 -u 0 -s 2000 -d 1';
z=0; Z=12;
vol_slice=;
cmd=$( get_cmd_prob2mask )
single_job "$cmd"
h52nii '' $dataset '_masks' 'maskMM_testfilter' '' '' '-i zyx -o xyz'


ipf='_masks'; ids='maskMM';
opf='_masks'; ods='maskMM_testfilter';
arg='-g -l 0 -u 0 -s 2000 -d 1';
vol_slice=;
# for z in `seq 0 12 184`; do
for z in `seq 47 12 60`; do
echo $z
Z=$((z+12))
cmd=$( get_cmd_prob2mask )
single_job "$cmd"
done
h52nii '' $dataset '_masks' 'maskMM_testfilter' '' '' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'

blockreduce 'h' 6 1

h52nii '' $dataset '_masks' 'maskMM' '' '' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'



conncomp 'h' '2D' $dataset '_masks' 'maskMM' '_labels' 'labelMA_core2D'
h52nii '' $dataset '_labels' 'labelMA_core2D' '' '' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'

conncomp 'h' '2Dfilter' $dataset '_labels' 'labelMA_core2D' '_labels_mapall' '' # (1-2h on LOCALn=7 without maskMB)
# conncomp h '2Dprops' '_labels' 'labelMA_core2D' '_labels_mapall' ''
# props=( 'label' 'area' 'eccentricity' 'mean_intensity' \
#        'solidity' 'extent' 'euler_number' )
# for prop in ${props[@]}; do
#     h52nii '' '_labels_mapall' $prop '-i zyx -o xyz'
# done

# TODO: include scikit-learn classifier ipython notebook here
# jupyter nbconvert --to python 2Dprops_classification.ipynb
conncomp h '2Dto3D' '_labels' 'labelMA_core2D' '_labels' 'labelMA_3Dlabeled'














export PYTHONPATH=$scriptdir
datadir=$blockdir
dataset=B-NT-S10-2f_ROI_00_06480-07020_01480-02020_00000-00184
dataset=B-NT-S10-2f_ROI_00_06480-07020_01980-02520_00000-00184
h52nii '' $dataset '_probs_eed' 'sum0247_eed' '-i zyx -o xyz -u'



scontrol update jobid=1630421 partition=compute NumCPUs=14 TimeLimit=03:10:00



export PATH=:$PATH
source activate root
export PYTHONPATH=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/pyDM3reader
python /home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem/prob2mask.py /data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500/B-NT-S10-2f_ROI_00_00000-00520_00000-00520_00000-00184_probs_eed.h5/sum0247_eed /data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500/B-NT-S10-2f_ROI_00_00000-00520_00000-00520_00000-00184_masks.h5/maskMM -l 0.5 -s 2000 -d 1 -S



h52nii 'h' $dataset '' 'data' '-u -i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'h' $dataset '_probs_eed_sum16' 'sum16_eed' '-u -i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'h' $dataset '_masks' 'maskMM_raw' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'h' $dataset '_masks' 'maskMM' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'h' $dataset '_masks' 'maskICS' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'

h52nii '' $dataset_ds '_probs_eed' 'sum0247_eed' '' '' '-i zyx -o xyz -d float'
h52nii '' $dataset_ds '_probs_eed' 'sum16_eed' '' '' '-i zyx -o xyz -d float'





source activate root
export PYTHONPATH=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/pyDM3reader

import os
import h5py

datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
h5file = 'B-NT-S10-2f_ROI_00_probs_eed_sum16.h5'

h5path_file = os.path.join(datadir, h5file)
h5file = h5py.File(h5path_file, 'a')

# h5file['_probs_eed'][:,:,:1000] = h5file['sum16_eed'][:,:,:1000]
# del(h5file['sum16_eed'])
# h5file['sum16_eed'] = h5file['_probs_eed']
# del(h5file['_probs_eed'])

h5file.close()



import h5py
fpath = '/vols/Data/km/michielk/oxdata/P01/EM/M3/M3S1GNU/M3S1GNUds7_masks.h5'
h5file = h5py.File(fpath, 'a')
del(h5file['maskMA'])
h5file.close()







unset datastems
datastems=()
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_00980-01520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_01480-02020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_01980-02520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_02480-03020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_02980-03520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_03480-04020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_03980-04520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_04480-05020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_04980-05520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_05480-06020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_05980-06520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_06480-07020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00480-01020_06980-07520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00980-01520_06980-07520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_00980-01520_07480-08020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_00000-00520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_00480-01020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_00980-01520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_01480-02020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_01980-02520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_02480-03020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_02980-03520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_03480-04020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_03980-04520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_04480-05020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_04980-05520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_05480-06020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_01480-02020_05980-06520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_04480-05020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_04980-05520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_05480-06020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_05980-06520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_06480-07020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_06980-07520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03480-04020_07480-08020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_00000-00520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_00480-01020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_00980-01520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_01480-02020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_01980-02520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_02480-03020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_02980-03520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_03980-04520_03480-04020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_00980-01520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_01480-02020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_02980-03520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_03480-04020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_03980-04520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_04480-05020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_05480-06020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_05980-06520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_06480-07020_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_06980-07520_06980-07520_00000-00184 )
datastems+=( B-NT-S10-2f_ROI_00_07480-08020_00000-00520_00000-00184 )

split_blocks 'h' '' '_probs' 'volume/predictions' $bs

eed 'h' '' '_probs' 'volume/predictions' '_probs_eed' 'probs_eed' '10:10:00'
