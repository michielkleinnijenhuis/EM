# revisiting gala (0.4-dev)

conda create -n np_gala_ilastik -c flyem neuroproof -c ilastik ilastik-everything

conda create --name gala_20160715 python=3.5 pytest coverage pytest-cov numpy nose numpydoc pillow networkx h5py scipy cython scikit-image scikit-learn pyzmq
source activate gala_20160715
pip install viridis
pip install git+git://github.com/janelia-flyem/gala@master
pip install /Users/michielk/workspace/FlyEM/gala
# run
source activate gala_20160715

# in gala: changed ilastik_headless to
# source activate ilastik-devel_20160714
# CONDA_ROOT=`conda info --root`
# ${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug
# alias ilastik_headless="${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless"

cd ~/workspace/FlyEM/gala
gala-pixel --config-file example/pix-config.json testsession
# example/train_command
# gala-train  --use-neuroproof -I --seed-cc-threshold 5 -o ./train-sample --experiment-name agglom ./example/prediction.h5 ./example/groundtruth
gala-train  -I --seed-cc-threshold 5 -o ./train-sample --experiment-name agglom ./example/prediction.h5 ./example/groundtruth




scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; y=1000; xs=500; ys=500; z=30; Z=460;
[ $x == 5000 ] && X=5217 || X=$((x+xs))
[ $y == 4000 ] && Y=4460 || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

# this generates supervoxels, but badly (it gets the myelinated fibres, and the UA as one compartment)
gala-segmentation-pipeline \
--image-stack ${datadir}/${datastem}.h5 \
--ilp-file ${datadir}/pixprob_training.ilp \
--disable-gen-pixel \
--pixelprob-file ${datadir}/${datastem}_probs.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output ${datadir}/${datastem} \
--segmentation-thresholds 0.0
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}/supervoxels.h5 ${datadir}/${datastem}/supervoxels.nii.gz \
-i 'xyz' -l 'xyz' -e -0.0073 -0.0073 0.05

# gala-segmentation-pipeline \
# --image-stack ${datadir}/${datastem}.h5 \
# --ilp-file ${datadir}/pixprob_training.ilp \
# --disable-gen-pixel \
# --pixelprob-file ${datadir}/${datastem}_probs.h5 \
# --disable-gen-supervoxels \
# --supervoxels-file ${datadir}/${datastem}/supervoxels.h5 \
# --enable-gen-agglomeration \
# --enable-raveler-output \
# --enable-h5-output ${datadir}/${datastem} \
# --segmentation-thresholds 0.0

# python3 for gala_20160715; # python2.7 for gala
# imports
from gala import imio, classify, features, agglo, evaluate as ev
import os

datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dset_name = 'm000_01000-01500_01000-01500_00030-00460'
os.chdir(datadir)

# read in training data
gt_train = imio.read_h5_stack(dset_name + '_PA.h5')
pr_train = imio.read_h5_stack(dset_name + '_probs.h5', group='/volume/predictions')
# ws_train = imio.read_h5_stack(dset_name + '_slic_s00500_c2.000_o0.050.h5')
ws_train = imio.read_h5_stack(os.path.join(dset_name, 'supervoxels.h5')

gt_train = gt_train[100:200,100:200,100:200]
pr_train = pr_train[100:200,100:200,100:200,:]
ws_train = ws_train[100:200,100:200,100:200]

# create a feature manager
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])

# create graph and obtain a training dataset
g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
(X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc)[0]
y = y[:, 0] # gala has 3 truth labeling schemes, pick the first one
print((X.shape, y.shape)) # standard scikit-learn input format

# train a classifier, scikit-learn syntax
rf = classify.DefaultRandomForest().fit(X, y)
# a policy is the composition of a feature map and a classifier
learned_policy = agglo.classifier_probability(fc, rf)

# get the test data and make a RAG with the trained policy
pr_test, ws_test = (map(imio.read_h5_stack,
                        ['test-p1.lzf.h5', 'test-ws.lzf.h5']))
g_test = agglo.Rag(ws_test, pr_test, learned_policy, feature_manager=fc)
g_test.agglomerate(0.5) # best expected segmentation
seg_test1 = g_test.get_segmentation()

# the same approach works with a multi-channel probability map
p4_train = imio.read_h5_stack('train-p4.lzf.h5')
# note: the feature manager works transparently with multiple channels!
g_train4 = agglo.Rag(ws_train, p4_train, feature_manager=fc)
(X4, y4, w4, merges4) = g_train4.learn_agglomerate(gt_train, fc)[0]
y4 = y4[:, 0]
print((X4.shape, y4.shape))
rf4 = classify.DefaultRandomForest().fit(X4, y4)
learned_policy4 = agglo.classifier_probability(fc, rf4)
p4_test = imio.read_h5_stack('test-p4.lzf.h5')
g_test4 = agglo.Rag(ws_test, p4_test, learned_policy4, feature_manager=fc)
g_test4.agglomerate(0.5)
seg_test4 = g_test4.get_segmentation()

# gala allows implementation of other agglomerative algorithms, including
# the default, mean agglomeration
g_testm = agglo.Rag(ws_test, pr_test,
                    merge_priority_function=agglo.boundary_mean)
g_testm.agglomerate(0.5)
seg_testm = g_testm.get_segmentation()

# examine how well we did with either learning approach, or mean agglomeration
gt_test = imio.read_h5_stack('test-gt.lzf.h5')
import numpy as np
results = np.vstack((
    ev.split_vi(ws_test, gt_test),
    ev.split_vi(seg_testm, gt_test),
    ev.split_vi(seg_test1, gt_test),
    ev.split_vi(seg_test4, gt_test)
    ))

print(results)
