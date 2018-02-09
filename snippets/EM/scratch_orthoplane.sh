mkdir build
cd build
cmake ..
make install

-DCMAKE_INSTALL_PREFIX=myprefix


### QT5 ###
PATH=/Users/michielk/Qt/5.9.1/clang_64/bin:$PATH

QTDIR=/usr/share/qtX qmake --version
PATH=/Users/michielk/Qt/5.9.1/clang_64:$PATH
qmake --version

PATH="/Users/michielk/anaconda/bin:${PATH}"

/Users/michielk/Qt/5.9.1/clang_64/plugins/designer/libqglviewerplugin.dylib
/Users/michielk/workspace/libQGLViewer-2.7.0
/Users/michielk/workspace/libQGLViewer-2.7.0/QGLViewer
/Users/michielk/workspace/libQGLViewer-2.7.0/QGLViewer/QGLViewer.framework/Headers


-DCMAKE_PREFIX_PATH=/Users/michielk/workspace/libQGLViewer-2.7.0

-D
QGLVIEWER_INCLUDE_DIR=/Users/michielk/workspace/libQGLViewer-2.7.0


LIBS *= -L/path/to/lib -lQGLViewer



/Users/michielk/Qt/5.9.1/clang_64/lib
