#! /usr/bin/bash

# make version
python python/versionData.py

# split data set
python python/splitData.py

# remove the old parsed file if exist
if [ -f bin/main ]; then
    rm bin/main
fi

if [ -f CMakeCache.txt ]; then
    rm CMakeCache.txt
fi

if [ -f Makefile ]; then
    rm Makefile
fi

if [ -f cmake_install.cmake ]; then
    rm cmake_install.cmake
fi

if [ -d "CMakeFiles" ]; then
	rm -r CMakeFiles
fi

# parsed code
cmake .
make

# run the program
time mpirun -np 11 bin/main