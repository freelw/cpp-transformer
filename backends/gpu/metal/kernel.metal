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


kernel void tensor_add_kernel(
    device float* dst [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device const float* src2 [[buffer(2)]],
    device const int* shape[[buffer(4)]],
    device const int* strides_dst[[buffer(5)]],
    device const int* strides_src1[[buffer(6)]],
    device const int* strides_src2[[buffer(7)]],
    device const int* args[[buffer(8)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int dim = args[0];
    int length = args[1];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    }

    int tmp_length = length;
    int tmp_index = index;
    int offset_src1 = 0;
    int offset_src2 = 0;
    int offset_dst = 0;
    for (int j = 0; j < dim; ++j) {
        tmp_length /= shape[j];
        int cur_dim_index = tmp_index / tmp_length;
        offset_src1 += cur_dim_index * strides_src1[j];
        offset_src2 += cur_dim_index * strides_src2[j];
        offset_dst += cur_dim_index * strides_dst[j];
        tmp_index %= tmp_length;
    }
    dst[offset_dst] = src1[offset_src1] + src2[offset_src2];
}