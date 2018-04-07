
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib nbagg')

import os
import h5py
import numpy as np
from sklearn import svm
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing
from skimage.morphology import binary_dilation
from wmem import utils, stack2stack

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/M3S1GNUds7"

labelfile = 'M3S1GNUds7_labelMA_core2D_fw_nf_label.h5'
maskfile = 'M3S1GNUds7_maskMA_final.h5'
basename = os.path.join(datadir, 'M3S1GNUds7_labels_mapall')
props = ('label', 'area', 'eccentricity', 'mean_intensity', 'solidity', 'extent', 'euler_number')

labelpath = os.path.join(datadir, labelfile, 'stack')
maskpath = os.path.join(datadir, maskfile, 'stack')
# basename = os.path.join(datadir, 'M3S1GNUds7_labelMA_core2D_fw_nf_mapall')
gtpath = '{}_{}.npy'.format(basename, 'groundtruth')
predpath = '{}_{}.npy'.format(basename, 'pred')
clfpath = os.path.join(datadir, 'clf_test.pkl')
scalerpath = os.path.join(datadir, 'scaler_test.pkl')

float_formatter = lambda x: "%.02f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


# In[ ]:


# determine ground truth

## get 2D labels and mask of identified 3D MA compartment
f = h5py.File(os.path.join(datadir, labelfile), 'r')
labels = f['stack'][:]
m = h5py.File(os.path.join(datadir, maskfile), 'r')
mask = m['stack'][:].astype('bool')
m.close()
f.close()

## split the labels in MA and notMA
labelsALL = np.unique(labels)
maskdil = binary_dilation(mask)
labelsMA = np.unique(labels[maskdil])
labelsNOTMA = np.unique(labels[~maskdil])

## filter labels that are split between compartments
labelsTRUE = set(labelsMA) - set(labelsNOTMA)
labelsFALSE = set(labelsALL) - set(labelsMA)
print(len(labelsTRUE), len(labelsFALSE))

## generate final ground truth labels
y = np.zeros_like(labelsALL, dtype='bool')
for l in labelsTRUE:
    y[l] = True
y[0] = False
np.save(gtpath, y)


# In[ ]:


# map the groundtruth labels to a volume

h5file_in, ds_in, elsize, axlab = utils.h5_load(labelpath)

h5path_out = os.path.join(datadir, 'M3S1GNUds7_gt.h5')

h5file_out, ds_out = utils.h5_write(None, ds_in.shape, 'uint8',
                                    os.path.join(h5path_out, 'class0'),
                                    element_size_um=elsize,
                                    axislabels=axlab)
ds_out[:] = ~y[ds_in[:]]
h5file_out.close()

h5file_out, ds_out = utils.h5_write(None, ds_in.shape, 'uint8',
                                    os.path.join(h5path_out, 'class1'),
                                    element_size_um=elsize,
                                    axislabels=axlab)
ds_out[:] = y[ds_in[:]]
h5file_out.close()

h5file_in.close()

# convert to nifti
niipath_out = os.path.join(datadir, 'M3S1GNUds7_gt_class0.nii.gz')
_ = stack2stack.stack2stack(os.path.join(h5path_out, 'class0'), niipath_out, inlayout='zyx', outlayout='xyz')
niipath_out = os.path.join(datadir, 'M3S1GNUds7_gt_class1.nii.gz')
_ = stack2stack.stack2stack(os.path.join(h5path_out, 'class1'), niipath_out, inlayout='zyx', outlayout='xyz')


# In[ ]:


# load the ground truth labels and remove the background
y_train = np.load(gtpath)
y_train = y_train[1:]
print(np.sum(y_train), np.sum(~y_train))


# In[ ]:


# load the training data (minus the background label)
X_train = np.zeros([y_train.shape[0], len(props) - 1])
for i, propname in enumerate(props[1:]):
    nppath = '{}_{}.npy'.format(basename, propname)
    X_train[:,i] = np.load(nppath)[1:]


# In[ ]:


# scale the training data
print(np.mean(X_train, axis=0))
print(np.mean(X_train[y_train,:], axis=0))
print(np.mean(X_train[~y_train,:], axis=0))

# X_train_scaled = preprocessing.scale(X_train)
# scaler = preprocessing.StandardScaler().fit(X_train)
# print(scaler.mean_)
# print(scaler.scale_)
# X_train_scaled = scaler.transform(X_train)
scaler = preprocessing.MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, scalerpath)


# In[ ]:


# plot the training data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

plt.figure()
X0 = X_train_scaled[y_train,:]
X1 = X_train_scaled[~y_train,:]
print(X0.shape, X1.shape)

plt.scatter(X0[::100, 1], X0[::100, 2], c=[1,0,0],
            alpha=0.5, edgecolor='k',
            label="Class0")
plt.scatter(X1[::100, 1], X1[::100, 2], c=[0,0,1],
            alpha=0.5, edgecolor='k',
            label="Class1")


# In[ ]:


# fit the support vector classifier
clf = svm.SVC()
# clf.fit(X_train_scaled[::100], y_train[::100])
clf.fit(X_train_scaled, y_train)
joblib.dump(clf, clfpath)
joblib.dump(scaler, scalerpath)


# In[ ]:


scaler = joblib.load(scalerpath)
clf = joblib.load(clfpath)


# In[ ]:


# load and scale the test data
datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
basename = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_mapall')
nppath = '{}_{}.npy'.format(basename, props[0])
tmp = np.load(nppath)[1:]

X_test = np.zeros([tmp.shape[0], len(props) - 1])
for i, propname in enumerate(props[1:]):
    nppath = '{}_{}.npy'.format(basename, propname)
    X_test[:,i] = np.load(nppath)[1:]

X_test_scaled = scaler.transform(X_test)


# In[ ]:


# prediction on test data
pred = clf.predict(X_test_scaled)


# In[ ]:


# save the results

## reinsert the background label
fw = np.insert(pred, 0, [False])

## save the predicted labels
predpath = '{}_{}.npy'.format(basename, 'pred')
np.save(predpath, fw)


# In[ ]:


datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
basename = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_mapall')
predpath = '{}_{}.npy'.format(basename, 'pred')
fw = np.load(predpath)


# In[ ]:


## map to volume
h5path_in = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels.h5', 'labelMA_core2D')
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
a = ds_in[:]
h5file_in.close()

h5path_out = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels.h5', 'labelMA_core2D_prediction')
h5file_out, ds_out = utils.h5_write(None, a.shape, a.dtype,
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)

mask = fw[a]
a[~mask] = 0
ds_out[:] = a
h5file_out.close()


# In[ ]:


## convert to nifti
niipath_out = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_prediction.nii.gz')
_ = stack2stack.stack2stack(h5path_out, niipath_out)  # , inlayout='zyx', outlayout='xyz'


# In[ ]:


# print(pred)
# print(pred_gt)
pred_gt = y
a = np.logical_xor(pred, pred_gt)
print(len(a), np.sum(~a) / float(len(a)))


# In[ ]:


h5path_out = os.path.join(datadir, 'M3S1GNUds7_pred.h5', 'stack')
h5file_in, ds_in, elsize, axlab = utils.h5_load(labelpath)
h5file_out, ds_out = utils.h5_write(None, ds_in.shape, 'uint8',
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)

ds_out[:] = fw[ds_in[:]]

h5file_in.close()
h5file_out.close()

niipath_out = os.path.join(datadir, 'M3S1GNUds7_pred.nii.gz')
_ = stack2stack.stack2stack(h5path_out, niipath_out, inlayout='zyx', outlayout='xyz')

