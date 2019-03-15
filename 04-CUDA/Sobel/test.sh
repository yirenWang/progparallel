touch results1.txt


#!/bin/bash
for i in 8 16 32 48 64 
do
echo "#define BLOCKDIM_X $i" > include/parameters.cuh
echo "#define BLOCKDIM_Y $i" >> include/parameters.cuh

make clean; make;

echo "#define BLOCKDIM_X $i" >> results1.txt
echo "#define BLOCKDIM_Y $i" >> results1.txt

exe/sobel.exe images/Drone.pgm 50 >> results3.txt 
exe/sobel.exe images/Drone_huge.pgm 50 >> results3.txt 
exe/sobel.exe images/Carre.pgm 50 >> results3.txt

done

