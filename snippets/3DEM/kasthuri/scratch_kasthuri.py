



import os
import h5py
import numnpy as np

datadir = '/Users/michielk/oxdata/P01/EM/Kashturi11/cutout03'
cutout = '-default-hdf5-3-650_1850-1700_2500-1000_1400-ocpcutout.h5'

Lmojo = loadh5(datadir, 'kat11mojocylinder' + cutout, '/default/CUTOUT')
Lred = loadh5(datadir, 'kat11redcylinder' + cutout, '/default/CUTOUT')
Lgreen = loadh5(datadir, 'kat11greencylinder' + cutout, '/default/CUTOUT')
Lall = Lmojo + Lred + Lgreen

writeh5(Lmojo, datadir, 'Lmojo.h5', dtype='uint8')
writeh5(Lred, datadir, 'Lred.h5', dtype='uint8')
writeh5(Lgreen, datadir, 'Lgreen.h5', dtype='uint8')
writeh5(Lall, datadir, 'Lall.h5', dtype='uint8')




# Lclass = np.zeros_like(labeldict['NN'], dtype='S4')
# Lclass.fill('NNNN')
# for label in np.unique(labeldict['NN']):
#     # the first level
#     if label in compdict['A']:  # second level is U or M; if M third level is M or A (or S); fourth level is O or V
#         c1 = 'A'
#     elif label in compdict['D']:  # second level is O (or S)
#         c1 = 'D'
#     elif label in compdict['S']:  # second level could be V?
#         c1 = 'S'
#     else:
#         c1 = 'E'
#     # the second level
#     if label in compdict['M']:
#     elif label in compdict['O']:
#     elif label in compdict['V']:
#     else:
#     # the third level
#
#     Lclass[labeldict['NN']==n] = 'MM'
#




import os
import h5py
import numpy as np

datadir = '/Users/michielk/oxdata/P01/EM/Kashturi11/cutout03'

compdict = {}
compdict['DD'] = np.loadtxt(os.path.join(datadir, os.pardir, 'dendrites.txt'), dtype='int')
compdict['UA'] = np.loadtxt(os.path.join(datadir, os.pardir, 'axons.txt'), dtype='int')
compdict['MA'] = np.loadtxt(os.path.join(datadir, os.pardir, 'MA.txt'), dtype='int')
compdict['MM'] = np.loadtxt(os.path.join(datadir, os.pardir, 'MM.txt'), dtype='int')

label = 15
for labelclass, labels in compdict.items():
    if label in labels:
        break

labelclass




# parent objects: ECS, DD + NN, MM, UA
# child objects :
# -- MA   (MM)
# -- mito (DD+NN, MA, UA)
# -- vesi (UA)
# -- syna (DD+ECS+UA)





datadir = '/Users/michielk/oxdata/P01/EM/Kashturi11/cutout03'
cutout = '-default-hdf5-3-650_1850-1700_2500-1000_1400-ocpcutout.h5'
Lmojo = loadh5(datadir, 'kat11mojocylinder' + cutout, '/default/CUTOUT')
