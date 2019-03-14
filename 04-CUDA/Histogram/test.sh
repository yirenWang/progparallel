touch results2.txt


#!/bin/bash
for i in 5 6 7 8 9 10 11 12 13 14 15 16
do
echo "#define BLOCKDIM_X $(($i*2))" > include/parameters.cuh
echo "#define BLOCKDIM_Y $(($i*2))" >> include/parameters.cuh

make clean; make;

echo "#define BLOCKDIM_X $(($i*2))" >> results1.txt
echo "#define BLOCKDIM_Y $(($i*2))" >> results1.txt

exe/sobel.exe images/Carre.pgm 50 >> results1.txt

done

