scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$scriptdir

python $scriptdir/wmem/stack2stack.py \
M3S1GNUds7_labelMA_core2D_fw_nf_label.h5/stack \
M3S1GNUds7_labelMA_core2D_fw_nf_label.nii.gz -i 'zyx' -o 'xyz'

python $scriptdir/wmem/stack2stack.py \
M3S1GNUds7_labelMA_core2D_fw_nf_solidity.h5/stack \
M3S1GNUds7_labelMA_core2D_fw_nf_solidity.nii.gz -i 'zyx' -o 'xyz'

python $scriptdir/wmem/stack2stack.py \
M3S1GNUds7_gt_fwmap.h5/stack \
M3S1GNUds7_gt_fwmap.nii.gz -i 'zyx' -o 'xyz'

datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/M3S1GNUds7"
props=('label' 'area' 'eccentricity' 'solidity' 'extent' 'euler_number')
python $scriptdir/wmem/connected_components.py \
$datadir/M3S1GNUds7_labelMA_core2D_fw_nf_label.h5/stack \
$datadir/M3S1GNUds7_labelMA_core2D_fw_nf_mapall.h5 \
-m '2Dfilter' -d 0 \
-p ${props[@]}




import os
import h5py
import numpy as np
from sklearn import svm
import pickle
from sklearn.externals import joblib

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/M3S1GNUds7"
labelfile = 'M3S1GNUds7_labelMA_core2D_fw_nf_label.h5'
maskfile = 'M3S1GNUds7_maskMA_final.h5'
basename = os.path.join(datadir, 'M3S1GNUds7_labelMA_core2D_fw_nf_mapall')
props = ('label', 'area', 'eccentricity', 'solidity', 'extent', 'euler_number')

f = h5py.File(os.path.join(datadir, labelfile), 'r')
labels = f['stack'][:]
m = h5py.File(os.path.join(datadir, maskfile), 'r')
mask = m['stack'][:].astype('bool')
m.close()
f.close()

labelsALL = np.unique(labels)
labelsMA = np.unique(labels[mask])
y = np.zeros_like(labelsALL, dtype='bool')
for l in labelsMA:
    y[l] = True
y[0] = False
nppath = '{}_{}.npy'.format(basename, 'groundtruth')
np.save(nppath, y)

X = np.zeros([y.shape[0], len(props) - 1])
for i, propname in enumerate(props[1:]):
    nppath = '{}_{}.npy'.format(basename, propname)
    X[:,i] = np.load(nppath)

# TODO: do the scaling...

clf = svm.SVC()
clf.fit(X, y)
clf.fit(X[1:100000], y[1:100000])
clf.predict([[2., 2., 2., 2., 2.]])

# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)
joblib.dump(clf, os.path.join(datadir, 'clf_test.pkl'))
# clf = joblib.load('filename.pkl')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

plt.figure()
X0 = X[y,:]
X1 = X[~y,:]

plt.scatter(X0[:, 0], X0[:, 1], c=[1,0,0],  # s=this_sw * 50,
            alpha=0.5, edgecolor='k',
            label="Class0")
plt.scatter(X1[:, 0], X1[:, 1], c=[0,0,1],  # s=this_sw * 50,
            alpha=0.5, edgecolor='k',
            label="Class1")

# for this_y, color in zip(y_unique, colors):
#     this_X = X_train[y_train == this_y]
#     this_sw = sw_train[y_train == this_y]
#     plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50, c=color,
#                 alpha=0.5, edgecolor='k',
#                 label="Class %s" % this_y)
# plt.legend(loc="best")
# plt.title("Data")
