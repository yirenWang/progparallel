#!/bin/bash
make clean; make

# for I in {4..127..1}
# do
# (( ITER=1000000))
# exe/SortieDeCache.exe $I $I $ITER
# done

# for I in {128..255..1}
# do
# (( ITER=10000))
# exe/SortieDeCache.exe $I $I $ITER
# done

# for I in {256..1023..1}
# do
# (( ITER=500))
# exe/SortieDeCache.exe $I $I $ITER
# done

# for I in {1024..2047..1}
# do
# (( ITER=10))
# exe/SortieDeCache.exe $I $I $ITER
# done

# for I in {2048..4095..1}
# do
# (( ITER=7))
# exe/SortieDeCache.exe $I $I $ITER
# done

# for I in {4096..8191..1}
# do
# (( ITER=4))
# exe/SortieDeCache.exe $I $I $ITER
# done

for I in {6958..8191..1}
do
(( ITER=3))
exe/SortieDeCache.exe $I $I $ITER
done

# for I in {8192..262144..4096}
# do
# (( ITER=3))
# exe/SortieDeCache.exe $I $I $ITER
# done

