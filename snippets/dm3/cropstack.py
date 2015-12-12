from ij import IJ
for imno in range(0,3):
	image = IJ.openImage('/Users/michielk/oxdata/originaldata/P01/EM/M2/Brain/I/26Sep14/img000-099/I 25Sep14_3VBSED_slice_' + imno.zfill(4) + '.dm3')

