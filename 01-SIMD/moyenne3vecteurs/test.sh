#!/bin/bash
make clean; make

(( ITER=1000))
exe/test.exe 1048576 $ITER
