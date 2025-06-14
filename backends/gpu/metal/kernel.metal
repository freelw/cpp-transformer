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

kernel void tensor_div_scalar_tensor(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const int* intArgs [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int length = intArgs[0];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    }
    else {
        dst[index] = src[index] / value[0];
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

kernel void tensor_l2_norm(
    device const float* Md [[buffer(0)]],
    device float* Nd [[buffer(1)]],
    device const int* args [[buffer(2)]],
    threadgroup float *partial_sums [[threadgroup(0)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = args[0];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    partial_sums[tid] = 0.0f;

    if (row >= M) {
        return;
    }
    else {
        partial_sums[tid] = pow(Md[row], 2);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device atomic_float* atomicData = (device atomic_float*)&Nd[0];
        atomic_fetch_add_explicit(atomicData, partial_sums[0], memory_order_relaxed);
    }
}

kernel void tensor_clip(
    device float* Md [[buffer(0)]],
    device const float* Norm [[buffer(1)]],
    device const int* intArgs [[buffer(2)]],
    device const float* floatArgs  [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = intArgs[0];
    float clip_value = floatArgs[0];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    }
    else {
        float norm = sqrt(Norm[0]);
        if (norm > clip_value) {
            Md[index] *= clip_value / norm;
        }
    }
}

kernel void tensor_adam_step(
    device float* w [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    device const int* intArgs [[buffer(4)]],
    device const float* floatArgs [[buffer(5)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = intArgs[0];
    int t = intArgs[1];
    float beta1 = floatArgs[0];
    float beta2 = floatArgs[1];
    float lr = floatArgs[2];
    float eps = floatArgs[3];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    }
    else {
        float w_value = w[index];
        float m_value = m[index];
        float v_value = v[index];
        float grad_value = grad[index];

        m_value = beta1 * m_value + (1.0f - beta1) * grad_value;
        v_value = beta2 * v_value + (1.0f - beta2) * pow(grad_value, 2);
        m[index] = m_value;
        v[index] = v_value;
        float m_hat = m_value / (1.0f - pow(beta1, t));
        float v_hat = v_value / (1.0f - pow(beta2, t));
        w_value -= lr * m_hat / (sqrt(v_hat) + eps);
        w[index] = w_value;
    }
}

kernel void reshape_deep_cp_float_kernel(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const int* src_shape [[buffer(2)]],
    device const int* src_strides [[buffer(3)]],
    device const int* args [[buffer(4)]],
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
        int offset = 0;
        for (int j = 0; j < dim; ++j) {
            tmp_length /= src_shape[j];
            int cur_dim_index = tmp_index / tmp_length;
            offset += cur_dim_index * src_strides[j];
            tmp_index %= tmp_length;
        }
        dst[index] = src[offset];
    }
}

kernel void repeat_interleave_int32_kernel(
    device const int32_t* src [[buffer(0)]],
    device int32_t* dst [[buffer(1)]],
    device const int* args [[buffer(2)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int32_t width = args[0];
    int32_t src_length = args[1];
    int32_t dst_length = args[2];
    int32_t n = args[3];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= dst_length) {
        return;
    }
    else {
        int j = index / (width * n);
        int k = index % width;
        int offset = j * width + k;
        dst[index] = src[offset];
    }
}

kernel void sequence_mask_kernel(
    device const float* src [[buffer(0)]],
    device const int* mask [[buffer(1)]],
    device float* dst [[buffer(2)]],
    device const int* intArgs [[buffer(3)]],
    device const float* floatArgs [[buffer(4)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int M = intArgs[0];
    int N = intArgs[1];
    int l_stride0 = intArgs[2];
    int l_stride1 = intArgs[3];
    int m_stride0 = intArgs[4];
    int r_stride0 = intArgs[5];
    int r_stride1 = intArgs[6];
    float value = floatArgs[0];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    else {
        int index_l = row * l_stride0 + col * l_stride1;
        int index_m = row * m_stride0;
        int index_r = row * r_stride0 + col * r_stride1;
        dst[index_r] = mask[index_m] <= col ? value : src[index_l];
    }
}

kernel void softmax_kernel(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device int* args [[buffer(2)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int shape0 = args[0];
    int shape1 = args[1];
    int shape2 = args[2];
    int l_stride0 = args[3];
    int l_stride1 = args[4];
    int l_stride2 = args[5];
    int r_stride0 = args[6];
    int r_stride1 = args[7];
    int r_stride2 = args[8];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= shape0 || col >= shape1) {
        return;
    }
    else {
        float max = -1e10;
        for (int i = 0; i < shape2; ++i) {
            float val = src[row * l_stride0 + col * l_stride1 + i * l_stride2];
            max = fmax(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < shape2; ++i) {
            float val = src[row * l_stride0 + col * l_stride1 + i * l_stride2];
            sum += exp(val - max);
        }
        for (int i = 0; i < shape2; ++i) {
            float val = src[row * l_stride0 + col * l_stride1 + i * l_stride2];
            dst[row * r_stride0 + col * r_stride1 + i * r_stride2] = exp(val - max) / sum;
        }
    }
}

kernel void softmax_backward_kernel(
    device float* target_grad [[buffer(0)]],
    device const float* softmax_res [[buffer(1)]],
    device const float* grad [[buffer(2)]],
    device const int* args [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int shape0 = args[0];
    int shape1 = args[1];
    int shape2 = args[2];
    int t_stride0 = args[3];
    int t_stride1 = args[4];
    int t_stride2 = args[5];
    int s_stride0 = args[6];
    int s_stride1 = args[7];
    int s_stride2 = args[8];
    int g_stride0 = args[9];
    int g_stride1 = args[10];
    int g_stride2 = args[11];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= shape0 || col >= shape1) {
        return;
    }
    else {
        for (int target = 0; target < shape2; ++target) {
            int tg_target_pos = row * t_stride0 + col * t_stride1 + target * t_stride2;
            float tmp = 0;
            for (int k = 0; k < shape2; ++k) {
                // int tg_k_pos = row * t_stride0 + col * t_stride1 + k * t_stride2;
                int sm_target_pos = row * s_stride0 + col * s_stride1 + target * s_stride2;
                int sm_k_pos = row * s_stride0 + col * s_stride1 + k * s_stride2;
                // int g_target_pos = row * g_stride0 + col * g_stride1 + target * g_stride2;
                int g_k_pos = row * g_stride0 + col * g_stride1 + k * g_stride2;

                float softmax_res_target = softmax_res[sm_target_pos];
                float softmax_res_k = softmax_res[sm_k_pos];
                float grad_k = grad[g_k_pos];
                tmp += (target == k ? softmax_res_k * (1 - softmax_res_k) : -softmax_res_target * softmax_res_k) * grad_k;
            }
            target_grad[tg_target_pos] = tmp;
        }
    }
}

kernel void tensor_embedding_kernel(
    device float* dst [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* src [[buffer(2)]],
    device const int* args [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int src_shape0 = args[0];
    int src_shape1 = args[1];
    int length = args[2];
    int src_stride0 = args[3];
    int src_stride1 = args[4];
    int dst_stride0 = args[5];
    int dst_stride1 = args[6];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= length || col >= src_shape1) {
        return;
    }
    else {
        int index_src = indices[row] * src_stride0 + col * src_stride1;
        int index_dst = row * dst_stride0 + col * dst_stride1;
        dst[index_dst] = src[index_src];
    }
}

kernel void tensor_embedding_backward_kernel(
    device float* dst [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* src [[buffer(2)]],
    device const int* args [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int src_shape0 = args[0];
    int src_shape1 = args[1];
    int length = args[2];
    int src_stride0 = args[3];
    int src_stride1 = args[4];
    int dst_stride0 = args[5];
    int dst_stride1 = args[6];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= length || col >= src_shape1) {
        return;
    }
    else {
        int index_src = row * src_stride0 + col * src_stride1;
        int index_dst = indices[row] * dst_stride0 + col * dst_stride1;
        device atomic_float* atomicData = (device atomic_float*)&dst[index_dst];
        atomic_fetch_add_explicit(atomicData, src[index_src], memory_order_relaxed);
    }
}

kernel void tensor_sum_2d_dim1(
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
    int tid = threadIdx.y * blockDim.y + threadIdx.x;
    partial_sums[tid] = 0.0f;
    if (row >= src_shape0 || col >= src_shape1) {
        return;
    } else {
        partial_sums[tid] = src[row * src_stride0 + col * src_stride1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sums[tid] += partial_sums[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            device atomic_float* atomicData = (device atomic_float*)&sum[row * sum_stride0];
            atomic_fetch_add_explicit(atomicData, partial_sums[0], memory_order_relaxed);
        }
    }
}

kernel void tensor_var_2d_dim1(
    device const float* src [[buffer(0)]],
    device const float* avg [[buffer(1)]],
    device float* sum [[buffer(2)]],
    device const int* args [[buffer(3)]],
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
    int tid = threadIdx.y * blockDim.y + threadIdx.x;
    partial_sums[tid] = 0.0f;
    if (row >= src_shape0 || col >= src_shape1) {
        return;
    } else {
        float _avg = avg[row * sum_stride0];
        float _src = src[row * src_stride0 + col * src_stride1];
        float diff = _src - _avg;
        partial_sums[tid] = pow(diff, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sums[tid] += partial_sums[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            device atomic_float* atomicData = (device atomic_float*)&sum[row * sum_stride0];
            atomic_fetch_add_explicit(atomicData, partial_sums[0], memory_order_relaxed);
        }
    }
}

kernel void tensor_norm_kernel(
    device const float* src [[buffer(0)]],
    device const float* avg [[buffer(1)]],
    device const float* var [[buffer(2)]],
    device float* dst [[buffer(3)]],
    device const int* args [[buffer(4)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int src_shape0 = args[0];
    int src_shape1 = args[1];
    int src_stride0 = args[2];
    int src_stride1 = args[3];
    int dst_stride0 = args[4];
    int dst_stride1 = args[5];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const float eps = 1e-5f;
    if (row >= src_shape0 || col >= src_shape1) {
        return;
    } else {
        float _avg = avg[row];
        float _var = var[row];
        float _src = src[row * src_stride0 + col * src_stride1];
        dst[row * dst_stride0 + col * dst_stride1] =
            (_src - _avg) / sqrt(_var + eps);
    }
}

kernel void tensor_norm_backward_kernel(
    device const float* src,
    device const float* norm,
    device const float* var,
    device float* tgt,
    device const int* args,
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]]
) {
    int src_shape0 = args[0];
    int src_shape1 = args[1];
    int src_stride0 = args[2];
    int src_stride1 = args[3];
    int norm_stride0 = args[4];
    int norm_stride1 = args[5];
    int tgt_stride0 = args[6];
    int tgt_stride1 = args[7];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float eps = 1e-5f;
    if (row >= src_shape0 || i >= src_shape1) {
        return;
    } else {
        float tmp = 0;
        float var_value = var[row];
        for (int j = 0; j < src_shape1; ++j) {
            int eq = i == j;
            auto sigma = sqrtf(var_value + eps);
            auto x_hat_i = norm[row * norm_stride0 + i * norm_stride1];
            auto x_hat_j = norm[row * norm_stride0 + j * norm_stride1];
            auto part1 = eq * src_shape1 - 1 - x_hat_i * x_hat_j;
            auto part2 = src_shape1 * sigma;
            auto g = part1 / part2;
            tmp += g * src[row * src_stride0 + j * src_stride1];
        }
        tgt[row * tgt_stride0 + i * tgt_stride1] = tmp;
    }
}