from ij import IJ

inputdir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs'
outputdir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/stitched/'
n_slices = 2

for slc in range(0, n_slices):
	IJ.run("Grid/Collection stitching", "type=[Grid: row-by-row] order=[Right & Down                ] grid_size_x=2 grid_size_y=2 tile_overlap=10 first_file_index_i=0 directory=/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs file_names=0001_m{iii}.tif output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]");
	IJ.saveAs("Tiff", "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/0001_fused.tif");
#
#	run("Grid/Collection stitching", "type=[Grid: row-by-row] order=[Right & Down                ] grid_size_x=2 grid_size_y=2 tile_overlap=10 first_file_index_i=0 directory=" + inputdir " file_names=" + slc + "_m{iii}.tif output_textfile_name=" + slc + "_TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap subpixel_accuracy computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]);
#	saveAs("Tiff", outputdir + slc + "_fused.tif");
