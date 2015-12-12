import numpy as np
import urllib3
import zlib
from io import StringIO

http = urllib3.PoolManager()
try:
    f = http.request('GET', "http://openconnecto.me/ocp/ca/kasthuri11/npz/1/4000,4200/4000,4200/100,300/")
except URLError:
    assert 0

zdata = f.read()

datastr = zlib.decompress(zdata[:])
datafobj = StringIO.StringIO(datastr)
cube = np.load ( datafobj )
cube.size
