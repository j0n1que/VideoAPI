#!/bin/bash

mkdir build
cd build
cmake ..
make
cp libinfer.so ../

