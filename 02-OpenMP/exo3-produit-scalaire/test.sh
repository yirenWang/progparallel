#!/bin/bash
make clean; make

(( ITER=10))
exe/test.exe 1048576 $ITER
