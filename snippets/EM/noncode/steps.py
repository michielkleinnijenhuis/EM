convert to tif (mpi)
(downsample) (mpi)
register
(condsider bias field correction)
(downsample) (mpi)
convert to h5
split in blocks
train ilastik classifier (manual)
apply ilastik classifier (on blocks [array] or fullstack)
EED (array)
masks (NOTE: size filter should be done on fullstack?)
merge blocks
blockreduce
2D connected components in maskMM
...
