
# coding: utf-8

# In[3]:


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

datadir_train = "/Users/michielk/oxdata/P01/EM/M3/M3S1GNU"
dataset_train = 'M3S1GNUds7'

clfpath = os.path.join(datadir_train, '{}_clf.pkl'.format(dataset_train))
scalerpath = os.path.join(datadir_train, '{}_scaler.pkl'.format(dataset_train))

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
dataset = 'B-NT-S10-2f_ROI_00ds7'
basename = os.path.join(datadir, '{}_labels_mapall'.format(dataset))

labelfile = '{}_labels.h5'.format(dataset)
labeldset = 'labelMA_core2D'
labelpath = os.path.join(datadir, labelfile, labeldset)

props = ('label', 'area', 'eccentricity', 'mean_intensity',
         'solidity', 'extent', 'euler_number')

float_formatter = lambda x: "%.02f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


# In[17]:


scaler = joblib.load(scalerpath)
clf = joblib.load(clfpath)


# In[19]:


# load and scale the test data

nppath = '{}_{}.npy'.format(basename, props[0])
tmp = np.load(nppath)[1:]

X_test = np.zeros([tmp.shape[0], len(props) - 1])
for i, propname in enumerate(props[1:]):
    nppath = '{}_{}.npy'.format(basename, propname)
    X_test[:,i] = np.load(nppath)[1:]

X_test_scaled = scaler.transform(X_test)


# In[20]:


# prediction on test data
pred = clf.predict(X_test_scaled)


# In[21]:


# save the results

## reinsert the background label
fw = np.insert(pred, 0, [False])

## save the predicted labels
predpath = '{}_{}.npy'.format(basename, 'pred')
np.save(predpath, fw)


# In[22]:


# load the prediction and apply additional criteria

predpath = '{}_{}.npy'.format(basename, 'pred')
fw = np.load(predpath)

propname = 'label'
nppath = '{}_{}.npy'.format(basename, propname)
fw_all = np.load(nppath)

propname = 'mean_intensity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
m = fwf > 0.8  # always include
fw[m] = fw_all[m]

propname = 'area'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
m = fwf > 3000  # always exclude
fw[m] = 0


# In[35]:


## map to volume

h5file_in, ds_in, elsize, axlab = utils.h5_load(labelpath)
a = ds_in[:]
h5file_in.close()

# myelinated axon labels
h5path_out = os.path.join(labelfile, '{}_prediction'.format(labeldset))
h5file_out, ds_out = utils.h5_write(None, a.shape, a.dtype,
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)

mask = fw[a]
a[~mask] = 0
ds_out[:] = a
h5file_out.close()

# myelinated axon mask
h5path_out = os.path.join(datadir, '{}_masks_maskPRED.h5'.format(dataset), 'maskMA')
h5file_out, ds_out = utils.h5_write(None, a.shape, 'bool',
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)
ds_out[:] = a.astype('bool')
h5file_out.close()


# In[36]:


## convert to nifti
niipath_out = os.path.join(datadir, '{}_labels_{}_prediction.nii.gz'.format(dataset, labeldset))
_ = stack2stack.stack2stack(h5path_out, niipath_out, inlayout='zyx', outlayout='xyz')

niipath_out = os.path.join(datadir, '{}_masks_maskPRED_maskMA.nii.gz'.format(dataset))
_ = stack2stack.stack2stack(h5path_out, niipath_out, inlayout='zyx', outlayout='xyz')

