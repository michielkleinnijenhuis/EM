import os
import numpy as np
import vtk
import h5py
from scipy import ndimage
from skimage import measure
import stl

datadir = '/Users/michielk/oxdata/P01/EM/Kashturi11/cutout01'
os.chdir(datadir)

x_start = 0
x_end = 500
y_start = 0
y_end = 500
z_start = 0
z_end = 100

f = h5py.File('kat11segments-default-hdf5-1-5000_5500-8000_8500-1100_1200-ocpcutout.h5', 'r')

label_im = f['default/CUTOUT'][z_start:z_end,y_start:y_end,x_start:x_end]
label_im = np.lib.pad(label_im.tolist(), ((1, 1), (1, 1), (1, 1)), 'constant')
# label_im, nb_labels = ndimage.label(f['default/CUTOUT'][z_start:z_end,y_start:y_end,x_start:x_end])
# label_im = measure.label(f['default/CUTOUT'][z_start:z_end,y_start:y_end,x_start:x_end])

labels = np.unique(label_im)
labels = np.delete(labels,0)
# nb_labels = len(labels)

spacing = (0.03, 0.006, 0.006)
xyzOffset = (1100, 8000, 5000)

for label in labels:
    labelmask = label_im == label
    verts, faces = measure.marching_cubes(labelmask, 0, spacing=spacing)
    faces = measure.correct_mesh_orientation(labelmask, verts, faces, spacing=spacing, gradient_direction='descent')
    # from scikit_image correct_mesh_orientation
    # Fancy indexing to define two vector arrays from triangle vertices
    actual_verts = verts[faces]
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    # Find normal vectors for each face via cross product
    crosses = np.cross(a, b)
    normals = crosses / (np.sum(crosses ** 2, axis=1) ** (0.5))[:, np.newaxis]
    ob = stl.Solid(label)
    for ii,face in enumerate(faces):
        ob.add_facet(normals[ii], actual_verts[ii])
    with open(str(label) + ".stl", 'w') as f:  #with open("allobjects.stl", 'a') as f:
        ob.write_binary(f)
        f.write("\n");








dims = label_im.shape

vol = vtk.vtkImageData()
vol.SetDimensions(dims[0], dims[1], dims[2])
vol.SetOrigin(xyzOffset[0]*spacing[0],xyzOffset[1]*spacing[1],xyzOffset[2]**spacing[2])  # vol.SetOrigin(0, 0, 0)
vol.SetSpacing(spacing[0], spacing[1], spacing[2])
sc = vtk.vtkFloatArray()
sc.SetNumberOfValues(label_im.size)
sc.SetNumberOfComponents(1)
sc.SetName('tnf')
for ii,tmp in enumerate(np.ravel(label_im.swapaxes(0,2))):
    sc.SetValue(ii,tmp)

vol.GetPointData().SetScalars(sc)

dmc = vtk.vtkDiscreteMarchingCubes()
dmc.SetInput(vol)
dmc.ComputeNormalsOn()

for ii,label in enumerate(labels):
    print("Processing labelnr: " + str(ii) + " with value: " + str(label))
    dmc.SetValue(0, label)  # dmc.SetValue(ii,tmp)
    # dmc.GenerateValues(nb_labels, 0, nb_labels)
    dmc.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(dmc.GetOutputPort())
    writer.SetFileName(str(label) + ".stl")
    writer.Write()
