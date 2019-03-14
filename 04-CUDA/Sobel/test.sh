touch results1.txt


#!/bin/bash
for i in 12 24 48 96 192
do
echo "#define BLOCKDIM_X $i" > include/parameters.cuh
echo "#define BLOCKDIM_Y $i" >> include/parameters.cuh

make clean; make;

echo "#define BLOCKDIM_X $i" >> results1.txt
echo "#define BLOCKDIM_Y $i" >> results1.txt

exe/sobel.exe images/Drone.pgm 50 >> results1.txt 
exe/sobel.exe images/Drone_huge.pgm 50 >> results1.txt 
exe/sobel.exe images/Carre.pgm 50 >> results1.txt

done

