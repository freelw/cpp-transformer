#include "kernel.cuh"
#include <stdio.h>

#ifndef GCC_CPU

__global__ void fill_float(
    float *Md, int M, float value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        Md[idx] = value;
    }
}

__global__ void tensor_add_2d(
    float *Md, float *Nd, float *Pd, //Pd is result
    int M, int N, // shape
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
    ) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    } else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_N = row * stride_N0 + col * stride_N1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] + Nd[index_N];
    }
}

__global__ void tensor_at_2d(
    float *Md, float *Nd, float *Pd,
    int M, int N, int P,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_Md[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_Nd[TILE_WIDTH][TILE_WIDTH];
    
    for (int m = 0; m < (N + TILE_WIDTH - 1)/ TILE_WIDTH; ++m) {
        // Load data into shared memory

        int M_row = row;
        int M_col = m * TILE_WIDTH + threadIdx.x;
        int N_row = m * TILE_WIDTH + threadIdx.y;
        int N_col = col;
        s_Md[threadIdx.y][threadIdx.x] =
            M_row < M && M_col < N ?
            Md[M_row * stride_M0 + M_col * stride_M1] : 0.f;
        s_Nd[threadIdx.y][threadIdx.x] =
            N_row < N && N_col < P?
            Nd[N_row * stride_N0 + N_col * stride_N1] : 0.f;
        __syncthreads();
        if (row >= M || col >= P) {
            
        } else {
            float sum = 0.0f;
            for (int k = 0; k < TILE_WIDTH; ++k) {
                sum += s_Md[threadIdx.y][k] * s_Nd[k][threadIdx.x];
            }
            Pd[row * stride_P0 + col * stride_P1] += sum;
        }
        __syncthreads();
    }
}

__global__ void tensor_add_eq_kernel(
    float *dst, float *src,
    int32_t *shape,
    int32_t *strides_dst,
    int32_t *strides_src,
    int32_t dim,
    int32_t length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
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

__global__ void expand_add(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_P0, int stride_P1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    } else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] + Nd[col];
    }
}

__global__ void expand_mul(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_P0, int stride_P1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    } else {
        int index_M = row * stride_M0 + col * stride_M1;
        int index_P = row * stride_P0 + col * stride_P1;
        Pd[index_P] = Md[index_M] * Nd[col];
    }
}

__global__ void tensor_mul_kernel(
    float *dst, float *src1, float *src2,
    int32_t *shape,
    int32_t *strides_dst,
    int32_t *strides_src1,
    int32_t *strides_src2,
    int32_t dim,
    int32_t length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
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

__global__ void tensor_add_kernel(
    float *dst, float *src1, float *src2,
    int32_t *shape,
    int32_t *strides_dst,
    int32_t *strides_src1,
    int32_t *strides_src2,
    int32_t dim,
    int32_t length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
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
}

__global__ void tensor_sum_2d_dim0(
    float *Md, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1
) {
    // todo: this kernel should be optimized
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) {
        return;
    } else {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += Md[i * stride_M0 + col * stride_M1];
        }
        Pd[col] = sum;
    }
}

__global__ void cross_entropy(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *loss,
    int M, int N,
    int stride0, int stride1
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    } else {
        float max = -1e10;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float val = Md[row * stride0 + i * stride1];
            max = fmaxf(max, val);
        }
        maxs[row] = max;
        for (int i = 0; i < N; ++i) {
            float val = Md[row * stride0 + i * stride1];
            sum += expf(val - max);
        }
        sums[row] = sum;
        int32_t label = labels[row];
        float zt = Md[row * stride0 + label * stride1];
        loss[row] = -zt + max + logf(sum);
    }
}

__global__ void cross_entropy_backward(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *grad,
    int M, int N,
    int Md_stride0, int Md_stride1,
    int grad_stride0, int grad_stride1
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    } else {
        float max = maxs[row];
        float sum = sums[row];
        int label = labels[row];
        for (int i = 0; i < N; ++i) {
            float val = Md[row * Md_stride0 + i * Md_stride1];
            grad[row * grad_stride0 + i * grad_stride1] = i == label ?
                (expf(val - max) / sum - 1) :
                expf(val - max) / sum;
        }
    }
}

__global__ void tensor_relu(
    float *Md, float *Nd, int M
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        Nd[index] = fmaxf(Md[index], 0.f);
    }
}

__global__ void tensor_relu_prime(
    float *Md, float *Nd, int M
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        Nd[index] = Md[index] > 0.f ? 1.f : 0.f;
    }
}

__global__ void tensor_l2_norm(
    float *Md, float *Nd, int M
) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    partial_sums[tid] = 0.0f;

    if (row >= M) {
        return;
    } else {
        partial_sums[tid] = powf(Md[row], 2);
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(Nd, partial_sums[0]);
    }
}

__global__ void tensor_clip(
    float *Md, float *Norm, int M,
    float clip_value
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        float norm = sqrtf(Norm[0]);
        if (norm > clip_value) {
            Md[index] *= clip_value / norm;
        }
    }
}

__global__ void tensor_adam_step(
    float *w, float *grad,
    float *m, float *v,
    int M, int t,
    float beta1, float beta2,
    float lr, float eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M) {
        return;
    } else {
        float w_value = w[index];
        float m_value = m[index];
        float v_value = v[index];
        float grad_value = grad[index];

        m_value = beta1 * m_value + (1.0f - beta1) * grad_value;
        v_value = beta2 * v_value + (1.0f - beta2) * powf(grad_value, 2);
        m[index] = m_value;
        v[index] = v_value;
        float m_hat = m_value / (1.0f - powf(beta1, t));
        float v_hat = v_value / (1.0f - powf(beta2, t));
        w_value -= lr * m_hat / (sqrtf(v_hat) + eps);
        w[index] = w_value;
    }
}

__global__ void reshape_deep_cp_float_kernel(
    float *dst, float *src,
    int32_t *src_shape, int32_t *src_strides,
    int32_t dim, int32_t length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
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

__global__ void repeat_interleave_int32_kernel(
    int32_t *src, int32_t *dst,
    int32_t width,
    int32_t src_length, int32_t dst_length,
    int32_t n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= dst_length) {
        return;
    } else {
        int j = index /(width * n);
        int k = index % width;
        int offset = j * width + k;
        dst[index] = src[offset];
    }
}

__global__ void sequence_mask_kernel(
    float *src, int32_t *mask, float *dst,
    int M, int N,
    int l_stride0,
    int l_stride1,
    int m_stride0,
    int r_stride0,
    int r_stride1,
    float value
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    } else {
        int index_l = row * l_stride0 + col * l_stride1;
        int index_m = row * m_stride0;
        int index_r = row * r_stride0 + col * r_stride1;
        dst[index_r] = mask[index_m] <= col ? value : src[index_l];
    }
}

__global__ void softmax_kernel(
    float *src, float *dst,
    int shape0, int shape1, int shape2,
    int l_stride0, int l_stride1, int l_stride2,
    int r_stride0, int r_stride1, int r_stride2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= shape0 || col >= shape1) {
        return;
    } else {
        float max = -1e10;
        for (int i = 0; i < shape2; ++i) {
            float val = src[row * l_stride0 + col * l_stride1 + i * l_stride2];
            max = fmaxf(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < shape2; ++i) {
            float val = src[row * l_stride0 + col * l_stride1 + i * l_stride2];
            sum += expf(val - max);
        }
        for (int i = 0; i < shape2; ++i) {
            float val = src[row * l_stride0 + col * l_stride1 + i * l_stride2];
            dst[row * r_stride0 + col * r_stride1 + i * r_stride2] = expf(val - max) / sum;  
        }
    }
}

__global__ void softmax_backward_kernel(
    float *target_grad, float *softmax_res, float *grad,
    int shape0, int shape1, int shape2,
    int t_stride0, int t_stride1, int t_stride2,
    int s_stride0, int s_stride1, int s_stride2,
    int g_stride0, int g_stride1, int g_stride2
) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= shape0 || col >= shape1) {
        return;
    } else {
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

__global__ void tensor_div_scalar(
    float *dst, float *src,
    int length, float value
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
        dst[index] = src[index] / value;
    }
}

__global__ void build_dropout_mask_kernel(
    float *mask,
    int32_t *shape,
    int32_t *strides,
    int length, int dim, float p
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
        int tmp_length = length;
        int tmp_index = index;
        int offset = 0;
        for (int j = 0; j < dim; ++j) {
            tmp_length /= shape[j];
            int cur_dim_index = tmp_index / tmp_length;
            offset += cur_dim_index * strides[j];
            tmp_index %= tmp_length;
        }
        mask[offset] = mask[offset] < p ? 0.0f : 1.0f;
    }
}

__global__ void tensor_embedding_kernel(
    float *dst,
    int32_t *indices,
    float *src,
    int src_shape0, int src_shape1,
    int length,
    int src_stride0, int src_strid1,
    int dst_stride0, int dst_stride1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= length || col >= src_shape1) {
        return;
    } else {
        int index_src = indices[row] * src_stride0 + col * src_strid1;
        int index_dst = row * dst_stride0 + col * dst_stride1;
        dst[index_dst] = src[index_src];
    }
}

__global__ void tensor_embedding_backward_kernel(
    float *dst,
    int32_t *indices,
    float *src,
    int src_shape0, int src_shape1,
    int length,
    int src_stride0, int src_strid1,
    int dst_stride0, int dst_stride1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= length || col >= src_shape1) {
        return;
    } else {
        int index_src = row * src_stride0 + col * src_strid1;
        int index_dst = indices[row] * dst_stride0 + col * dst_stride1;
        // dst[index_dst] += src[index_src];
        atomicAdd(&dst[index_dst], src[index_src]); // todo : atomicAdd is not efficient
        //printf("index_dst : %d, index_src : %d src val %f\n",
        //       index_dst, index_src, src[index_src]);
    }
}

__global__ void tensor_sum_2d_dim0_v1(
    float *src, float *sum,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int sum_stride0
) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    partial_sums[tid] = 0.0f;
    if (row >= src_shape0 || col >= src_shape1) {
        return;
    } else {
        partial_sums[tid] = src[row * src_stride0 + col * src_stride1];
        __syncthreads();
        for (int s = blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sums[tid] += partial_sums[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(&sum[col * sum_stride0], partial_sums[0]);
        }
    }
}

__global__ void tensor_sum_2d_dim1(
    float *src, float *sum,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int sum_stride0
) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.y + threadIdx.x;
    partial_sums[tid] = 0.0f;
    if (row >= src_shape0 || col >= src_shape1) {
        return;
    } else {
        partial_sums[tid] = src[row * src_stride0 + col * src_stride1];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sums[tid] += partial_sums[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(&sum[row * sum_stride0], partial_sums[0]);
        }
    }
}

__global__ void tensor_var_2d_dim1(
    float *src, float *avg, float *sum,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int sum_stride0
) {

    extern __shared__ float partial_sums[];
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
        partial_sums[tid] = powf(diff, 2);
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial_sums[tid] += partial_sums[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(&sum[row * sum_stride0], partial_sums[0]);
        }
    }
}

__global__ void tensor_norm_kernel(
    float *src, float *avg, float * var, float *dst,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int dst_stride0, int dst_stride1
) {
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
            (_src - _avg) / sqrtf(_var + eps);
    }
}

__global__ void tensor_norm_backward_kernel(
    float *src, float *norm, float * var, float *tgt,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int norm_stride0, int norm_stride1,
    int tgt_stride0, int tgt_stride1
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float eps = 1e-5f;
    if (row >= src_shape0 || i >= src_shape1) {
        return;
    } else {
        float tmp = 0;
        float var_value = var[row];
        for (int j= 0; j < src_shape1; ++ j) {
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

__global__ void tensor_mul_scalar(
    float *dst, float *src,
    int length, float value
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    } else {
        dst[index] = src[index] * value;
    }
}

#endif // GCC_CPU