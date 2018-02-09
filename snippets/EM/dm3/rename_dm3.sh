STEM=A-F+-T5-1D2-g4B
STEM=A-NT-T5-2B1-g5A
STEM=A-F+T3-1C2d-g6B
STEM=A-F+-T1-1C6e-g7C
STEM=A-F+-T3-1C2b-g8B
STEM=A-F+-T1-1C6b-g9A
STEM=A-F+-T1-1C6a-g10D
STEM=A-F+T3-1C2d-g6F

for f in `ls *OneView*.dm3`; do
PF=${f##*OneView_}
mv $f $STEM$PF
done
