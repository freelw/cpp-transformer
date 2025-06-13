#include <metal_stdlib>
using namespace metal;

#define TILE_WIDTH 32

kernel void fill_float(
    device float* Md [[buffer(0)]],
    device const int* argsInt[[buffer(1)]],
    device const float* argsFloat[[buffer(2)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = argsInt[0];
    float value = argsFloat[0];
    if (idx < M) {
        Md[idx] = value;
    }
}

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

kernel void tensor_at_2d(
    device const float* Md [[buffer(0)]],
    device const float* Nd [[buffer(1)]],
    device float* Pd [[buffer(2)]],
    device const int* args[[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    threadgroup float s_Md[TILE_WIDTH][TILE_WIDTH];
    threadgroup float s_Nd[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int M = args[0];
    int N = args[1];
    int P = args[2];
    int stride_M0 = args[3];
    int stride_M1 = args[4];
    int stride_N0 = args[5];
    int stride_N1 = args[6];
    int stride_P0 = args[7];
    int stride_P1 = args[8];

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load data into shared memory

        int M_row = row;
        int M_col = m * TILE_WIDTH + threadIdx.x;
        int N_row = m * TILE_WIDTH + threadIdx.y;
        int N_col = col;
        s_Md[threadIdx.y][threadIdx.x] =
            M_row < M && M_col < N ?
            Md[M_row * stride_M0 + M_col * stride_M1] : 0.f;
        s_Nd[threadIdx.y][threadIdx.x] =
            N_row < N && N_col < P ?
            Nd[N_row * stride_N0 + N_col * stride_N1] : 0.f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row >= M || col >= P) {

        }
        else {
            float sum = 0.0f;
            for (int k = 0; k < TILE_WIDTH; ++k) {
                sum += s_Md[threadIdx.y][k] * s_Nd[k][threadIdx.x];
            }
            Pd[row * stride_P0 + col * stride_P1] += sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void tensor_add_eq_kernel(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const int* shape [[buffer(2)]],
    device const int* strides_dst [[buffer(3)]],
    device const int* strides_src [[buffer(4)]],
    device const int* args [[buffer(5)]],
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
    else {
        int tmp_length = length;
        int tmp_index = index;
        int offset_src = 0;
        int offset_dst = 0;
        for (int j = 0; j < dim; ++j) {
            tmp_length /= shape[j];
            int cur_dim_index = tmp_index / tmp_length;
            offset_src += cur_dim_index * strides_src[j];
            offset_dst += cur_dim_index * strides_dst[j];
            tmp_index %= tmp_length;
        }
        dst[offset_dst] += src[offset_src];
    }
}

kernel void expand_add(
    device const float* Md [[buffer(0)]],
    device const float* Nd [[buffer(1)]],
    device float* Pd [[buffer(2)]],
    device const int* args[[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = args[0];
    int N = args[1];
    int stride_M0 = args[2];
    int stride_M1 = args[3];
    int stride_P0 = args[4];
    int stride_P1 = args[5];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] + Nd[col];
    }
}