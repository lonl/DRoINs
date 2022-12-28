#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda/

nvcc -std=c++11 -c -o roi_align_op.cu.o roi_align_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o roi_align.so roi_align_op.cc \
	roi_align_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
