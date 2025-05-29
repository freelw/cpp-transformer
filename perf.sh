#!/bin/bash

NCU_PATH="/usr/local/cuda/nsight-compute-2024.3.2/ncu"
NSIGHT_OUT_PATH="/home/cs/project/recognizing_handwritten_digits/c++/v5_tensor/nsight_out/"
mkdir -p ${NSIGHT_OUT_PATH}
${NCU_PATH} --set detailed \
-o ${NSIGHT_OUT_PATH}/mnist_cuda_out \
./mnist_cuda -g 1 -b 1 -e 5