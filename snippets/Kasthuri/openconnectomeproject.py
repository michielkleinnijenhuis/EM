import numpy as np
import urllib2
import zlib
import StringIO
import os

cut = 'kasthuri11/npz/1/4000,4200/4000,4200/100,300'

try:
	f = urllib2.urlopen ( os.path.join("http://openconnecto.me/ocp/ca/", cut) )
except URLError:
	assert 0

zdata = f.read ()

datastr = zlib.decompress ( zdata[:] )
datafobj = StringIO.StringIO ( datastr )
cube = np.load ( datafobj )
cube.size

filename = (cut.replace('/','_')).replace(',','-')
filepath = os.path.join(os.path.expanduser("~"), 'oxdata', 'fmrib', 'EM_OCP', filename)
np.save(filepath, cube)
