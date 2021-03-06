from __future__ import print_function

import vtk

import os 
os.chdir('/Users/michielk/oxdata/P01/EM/Kashturi11/downloads/kat11segments/hdf5/Labels')
input='labels.mhd'

# Prepare to read the file

readerVolume = vtk.vtkMetaImageReader()
readerVolume.SetFileName(input)
readerVolume.Update()


# Extract the region of interest
voi = vtk.vtkExtractVOI()
if vtk.VTK_MAJOR_VERSION <= 5:
    voi.SetInput(readerVolume.GetOutput())
else:
    voi.SetInputConnection(readerVolume.GetOutputPort())

voi.SetSampleRate(1,1,1)

voi.Update()

srange= voi.GetOutput().GetScalarRange()
print("Range", srange)


##Prepare surface generation
contour = vtk.vtkDiscreteMarchingCubes() #for label images
if vtk.VTK_MAJOR_VERSION <= 5:
    contour.SetInput( voi.GetOutput() )
else:
    contour.SetInputConnection( voi.GetOutputPort() )


#choose one label
index= 31

print("Doing label", index)

contour.SetValue(0, index)
contour.Update() #needed for GetNumberOfPolys() !!!


smoother= vtk.vtkWindowedSincPolyDataFilter()
if vtk.VTK_MAJOR_VERSION <= 5:
    smoother.SetInput(contour.GetOutput())
else:
    smoother.SetInputConnection(contour.GetOutputPort())

smoother.SetNumberOfIterations(30) #this has little effect on the error!
smoother.NonManifoldSmoothingOn()
smoother.NormalizeCoordinatesOn()
smoother.GenerateErrorScalarsOn()
smoother.Update()

smoothed_polys= smoother.GetOutput()
smoother_error= smoothed_polys.GetPointData().GetScalars()

##Find min and max z
se_range= smoother_error.GetRange()
print(se_range)
minz= se_range[0] #min(smoother_error)
maxz= se_range[1] #max(smoother_error)
if (maxz > 1):
    print("Big smoother error: min/max:", minz, maxz)

minz=  .3 #this way colours of different particles are comparable
maxz= 1

## Create the color map
colorLookupTable= vtk.vtkLookupTable()
colorLookupTable.SetTableRange(minz, maxz) #this does nothing, use mapper.SetScalarRange(minz, maxz)
colorLookupTable.SetHueRange(2/3.0, 1)
colorLookupTable.Build()

##calc cell normal
triangleCellNormals= vtk.vtkPolyDataNormals()
if vtk.VTK_MAJOR_VERSION <= 5:
    triangleCellNormals.SetInput(smoothed_polys)
else:
    triangleCellNormals.SetInputData(smoothed_polys)

triangleCellNormals.ComputeCellNormalsOn()
triangleCellNormals.ComputePointNormalsOff()
triangleCellNormals.ConsistencyOn()
triangleCellNormals.AutoOrientNormalsOn()
triangleCellNormals.Update() #creates vtkPolyData


mapper= vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(triangleCellNormals.GetOutput()) #this is better for vis;-)
else:
    mapper.SetInputConnection(triangleCellNormals.GetOutputPort()) #this is better for vis;-)

mapper.ScalarVisibilityOn()#show colour
mapper.SetScalarRange(minz, maxz)
#mapper.SetScalarModeToUseCellData() # contains the label eg. 31
mapper.SetScalarModeToUsePointData() #the smoother error relates to the verts
mapper.SetLookupTable(colorLookupTable)

# Take the isosurface data and create geometry
actor = vtk.vtkLODActor()
actor.SetNumberOfCloudPoints( 100000 )
actor.SetMapper(mapper)

# Create renderer
ren = vtk.vtkRenderer()
ren.SetBackground( 0, 0, 0 )
ren.AddActor(actor)

# Create a window for the renderer of size 250x250
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(250, 250)

# Set an user interface interactor for the render window
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Start the initialization and rendering
iren.Initialize()
renWin.Render()
iren.Start()
