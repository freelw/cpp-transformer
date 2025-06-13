#include <metal_stdlib>
using namespace metal;

kernel void tensor_add_2d(
    device const float* Md [[buffer(0)]],
    device const float* Nd [[buffer(1)]],
    device float* Pd [[buffer(2)]],
    device const int* args[[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int M = args[0];
    int N = args[1];

    int stride_M0 = args[2];
    int stride_M1 = args[3];
    int stride_N0 = args[4];
    int stride_N1 = args[5];
    int stride_P0 = args[6];
    int stride_P1 = args[7];

    if (row >= M || col >= N) {
        return;
    }
    else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_N = row * stride_N0 + col * stride_N1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] + Nd[index_N];
    }
}