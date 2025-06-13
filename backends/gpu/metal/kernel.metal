#include <metal_stdlib>
using namespace metal;

kernel void tensor_add_kernel(
    device float* dst [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device const float* src2 [[buffer(2)]],
    device const int* shape[[buffer(3)]],
    device const int* strides_dst[[buffer(4)]],
    device const int* strides_src1[[buffer(5)]],
    device const int* strides_src2[[buffer(6)]],
    device const int* args[[buffer(7)]],
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