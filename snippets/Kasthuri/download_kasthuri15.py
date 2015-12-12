# Import the OCPy access module to get data from the server
import ocpy.access

# Pick a token whose dataset you wish to download.
TOKEN = 'kasthuri11'
RESOLUTION = '1'
DATA_LOCATION = "downloads/" + TOKEN

# Find the bounds of the dataset using `get_info()`:
project_information = ocpy.access.get_info(TOKEN)
bounds = project_information['dataset']['imagesize'][RESOLUTION]
z_bounds = project_information['dataset']['slicerange']

# Before we download, let's store the starting directory so we can
# traverse the filesystem without worrying about getting lost:
import os
starting_directory = os.getcwd()

# Now let's download the dataset.
# The `get_data()` function returns a 2-tuple: (successes, failures).
# NOTE: This is going to take a
#                            ...REALLY
#                                  ...LONG
#                                      ...TIME
#                                           so go take a nap.
successes, failures = ocpy.access.get_data(
                                token=TOKEN,            resolution=RESOLUTION,
                                x_start=0,              x_stop=bounds[0],
                                y_start=0,              y_stop=bounds[1],
                                z_start=z_bounds[0],    z_stop=z_bounds[1],
                                location=DATA_LOCATION)

# Let's store our failed filenames in case we want to check these errors later
with open(DATA_LOCATION + '/errors.log', 'a') as f:
    for fail in failures:
        f.write(fail + "\n")

# Create a directory for exported PNG files:
os.mkdir(DATA_LOCATION + "/png")
os.chdir(DATA_LOCATION + "/hdf5")

# We'll need a few more libraries for this, including h5py and ocpy's
# Requests-handling and PNG-handling library:
from ocpy.convert import png
from ocpy.Request import *
import glob
import h5py

# Now convert the HDF5 files to PNG slices (2D, one per layer):
for filename in glob.glob("*.hdf5"):
    # First get the actual parameters from the HDF5 file.
    req = Request(filename)
    i = int(req.z_start)

    print("Slicing " + filename)
    f = h5py.File(filename, "r")
    # OCP stores data inside the 'cutout' h5 dataset
    data_layers = f.get('CUTOUT')

    out_files = []
    for layer in data_layers:
        # Filename is formatted like the request URL but `/` is `-`
        png_file = filename + "." + str(i).zfill(6) + ".png"

        out_files.append(
            png.export_png("../png/" + png_file, layer))
        i += 1
    # if you want, you have access to the out_files array here.

os.chdir(starting_directory)