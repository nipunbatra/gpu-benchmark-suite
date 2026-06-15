#!/bin/bash
wget -O gpu-burn.zip https://github.com/wilicc/gpu-burn/archive/refs/heads/master.zip
unzip gpu-burn.zip
cd gpu-burn-master
make
./gpu_burn 30