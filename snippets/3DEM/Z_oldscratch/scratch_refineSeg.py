







#         labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False)


MA = loadh5(datadir, dataset + MAfile)
# sesize = (5,5,5)
it = 10  # might be a bit agressive
for l in np.unique(MA)[1:]:
    ### fill holes
    # binary_fill_holes(binim, output=binim)  # does not fill components conneted to boundaries
    labels = label(MA!=l)[0]
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    MA[labels != background] = l
    ### closing
    # binary_closing(binim, structure=np.ones(sesize), iterations=it, output=binim)
    binim = MA==l
    binim = binary_closing(binim, iterations=it)
    MA[binim] = l
    ### fill holes
    labels = label(MA!=l)[0]
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    MA[labels != background] = l
    ### update myelin mask
    myelin[MA != 0] = False
    print(l)

writeh5(MA, datadir, dataset + '_probs_ws_MAfilled.h5')


import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
from scipy.special import expit

MM = loadh5(datadir, dataset + MMfile)
# distsum = np.zeros_like(MM, dtype='float')
distsum = np.ones_like(MM, dtype='float')
lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
medwidth = {}
for l in np.unique(MA)[1:]:
# for l in [1248,1249,1250,1352]:
    dist = distance_transform_edt(MA!=l, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
    # dist[MM!=l] = 0
    # get the median distance at the outer rim:
    MMfilled = MA+MM
    binim = MMfilled == l
    rim = np.logical_xor(erosion(binim), binim)
    medwidth = np.median(dist[rim])
    # nmed = 3
    # maxdist = nmed * medwidth  # np.histogram(dist, bins=100, density=True)
    # MM[np.logical_and(dist > maxdist, MM==l)] = 0
    # weighteddist = dist/medwidth
    weighteddist = expit(dist/medwidth)
    # weighteddist[weighteddist>1] = 1 + (weighteddist[weighteddist>1]-1) / 4
    # weighteddist[dist/medwidth>nmed] = 0
    # writeh5(weighteddist, datadir, dataset + '_probs_ws_wdist' + str(l) + '.h5', dtype='float')
    distsum = np.minimum(distsum, weighteddist)
    # distsum += weighteddist
    # writeh5(distsum, datadir, dataset + '_probs_ws_distsum' + str(l) + '.h5', dtype='float')
    # print(np.count_nonzero(distsum>nmed))
    # distsum[distsum>nmed] = distsum[distsum>nmed] / 2
    print(l)
writeh5(distsum, datadir, dataset + '_probs_ws_distsum.h5', dtype='float')

tmpdistsum = np.copy(distsum)
tmpdistsum[~myelin] = 0
MM = watershed(tmpdistsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5')



MM = loadh5(datadir, dataset + MMfile)
distsum = np.ones_like(MM, dtype='float')
lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
medwidth = {}
for i,l in enumerate(np.unique(MA)[1:]):  # TODO: implement mpi
    dist = distance_transform_edt(MA!=l, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
    # get the median distance at the outer rim:
    MMfilled = MA+MM
    binim = MMfilled == l
    rim = np.logical_xor(erosion(binim), binim)
    medwidth[l] = np.median(dist[rim])
    # median width weighted sigmoid transform on distance function
    weighteddist = expit(dist/medwidth[l])
    distsum = np.minimum(distsum, weighteddist)
    # labelmask for voxels further than nmed medians from the object
    nmed = 3
    maxdist = nmed * medwidth[l]
    lmask[:,:,:,i] = dist > maxdist
    print(l)

MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5')

for i,l in enumerate(np.unique(MA)[1:]):
    MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0

writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum_distfilter.h5')









scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU/devel"
dataset='m000'
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile '_probs_ws_MAfilled.h5' \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5' \
-n 5 -o 50 60 122 -s 100 4111 4235 -e 0.05 0.0073 0.0073 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100


dataset='m000_00000-01000_00000-01000_00000-00100'
dataset='m000_02000-03000_02000-03000_00000-00100'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs.h5" \
"${datadir}/${dataset}_probs.nii.gz" -i 'zyxc' -l 'xyzc' -e -0.0073 -0.0073 0.05 1 -f 'volume/predictions'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MA.h5" \
"${datadir}/${dataset}_probs_ws_MA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MM.h5" \
"${datadir}/${dataset}_probs_ws_MM.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_UA.h5" \
"${datadir}/${dataset}_probs_ws_UA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_PA.h5" \
"${datadir}/${dataset}_probs_ws_PA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

for i in 0 1 2; do
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs${i}_eed2.h5" \
"${datadir}/${dataset}_probs${i}_eed2.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done

python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_distance.h5" \
"${datadir}/${dataset}_distance.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MAfilled.h5" \
"${datadir}/${dataset}_probs_ws_MAfilled.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_distsum.h5" \
"${datadir}/${dataset}_probs_ws_distsum.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MMdistsum.h5" \
"${datadir}/${dataset}_probs_ws_MMdistsum.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MMdistsum_distfilter.h5" \
"${datadir}/${dataset}_probs_ws_MMdistsum_distfilter.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
