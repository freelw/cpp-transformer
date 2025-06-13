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

kernel void tensor_mul_kernel(
    device float* dst [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device const float* src2 [[buffer(2)]],
    device const int* shape [[buffer(3)]],
    device const int* strides_dst [[buffer(4)]],
    device const int* strides_src1 [[buffer(5)]],
    device const int* strides_src2 [[buffer(6)]],
    device const int* args [[buffer(7)]],
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
        dst[offset_dst] = src1[offset_src1] * src2[offset_src2];
    }
}

kernel void tensor_sum_2d_dim0_v1(
    device const float* src [[buffer(0)]],
    device float* sum [[buffer(1)]],
    device const int* args [[buffer(2)]],
    threadgroup float *partial_sums [[threadgroup(0)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int src_shape0 = args[0];
    int src_shape1 = args[1];
    int src_stride0 = args[2];
    int src_stride1 = args[3];
    int sum_stride0 = args[4];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    partial_sums[tid] = 0.0f;
    if (row >= src_shape0 || col >= src_shape1) {
        return;
    }
    else {
        partial_sums[tid] = src[row * src_stride0 + col * src_stride1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sums[tid] += partial_sums[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            device atomic_float* atomicData = (device atomic_float*)&sum[col * sum_stride0];
            atomic_fetch_add_explicit(atomicData, partial_sums[0], memory_order_relaxed);
        }
    }
}

kernel void cross_entropy(
    device const float* Md [[buffer(0)]],
    device const int* labels [[buffer(1)]],
    device float* maxs [[buffer(2)]],
    device float* sums [[buffer(3)]],
    device float* loss [[buffer(4)]],
    device int* args [[buffer(5)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = args[0];
    int N = args[1];
    int stride0 = args[2];
    int stride1 = args[3];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    }
    else {
        float max = -1e10;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float val = Md[row * stride0 + i * stride1];
            max = fmax(max, val);
        }
        maxs[row] = max;
        for (int i = 0; i < N; ++i) {
            float val = Md[row * stride0 + i * stride1];
            sum += exp(val - max);
        }
        sums[row] = sum;
        int32_t label = labels[row];
        float zt = Md[row * stride0 + label * stride1];
        loss[row] = -zt + max + log(sum);
    }
}

kernel void cross_entropy_backward(
    device const float* Md [[buffer(0)]],
    device const int32_t* labels [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device const float* sums [[buffer(3)]],
    device float* grad [[buffer(4)]],
    device const int* args [[buffer(5)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = args[0];
    int N = args[1];
    int Md_stride0 = args[2];
    int Md_stride1 = args[3];
    int grad_stride0 = args[4];
    int grad_stride1 = args[5];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    }
    else {
        float max = maxs[row];
        float sum = sums[row];
        int label = labels[row];
        for (int i = 0; i < N; ++i) {
            float val = Md[row * Md_stride0 + i * Md_stride1];
            grad[row * grad_stride0 + i * grad_stride1] = i == label ?
                (exp(val - max) / sum - 1) :
                exp(val - max) / sum;
        }
    }
}

kernel void tensor_relu(
    device const float* Md [[buffer(0)]],
    device float* Nd [[buffer(1)]],
    device const int* args [[buffer(2)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = args[0];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    }
    else {
        Nd[index] = fmax(Md[index], 0.f);
    }
}

kernel void tensor_div_scalar(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const int* intArgs [[buffer(2)]],
    device const float* floatArgs [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int length = intArgs[0];
    float value = floatArgs[0];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    }
    else {
        dst[index] = src[index] / value;
    }
}

kernel void expand_mul(
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
        Pd[index_P] = Md[index_M] * Nd[col];
    }
}

kernel void tensor_relu_prime(
    device const float* Md [[buffer(0)]], 
    device float* Nd [[buffer(1)]],
    device const int* args [[buffer(2)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = args[0];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    }
    else {
        Nd[index] = Md[index] > 0.f ? 1.f : 0.f;
    }
}
