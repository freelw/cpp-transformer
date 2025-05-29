#ifndef V5_TENSOR_BACKENDS_GPU_KERNEL_CUH
#define V5_TENSOR_BACKENDS_GPU_KERNEL_CUH

#ifndef GCC_ASAN
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_WIDTH = 32;

__global__ void fill_float(
    float *Md, int M, float value
);

__global__ void tensor_add_2d(
    float *Md, float *Nd, float *Pd, //Pd is result
    int M, int N, // shape
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
);

__global__ void tensor_at_2d(
    float *Md, float *Nd, float *Pd,
    int M, int N, int P,
    int stride_M0, int stride_M1,
    int stride_N0, int stride_N1,
    int stride_P0, int stride_P1
);

__global__ void tensor_add_eq_kernel(
    float *dst, float *src,
    int32_t *shape,
    int32_t *strides_dst,
    int32_t *strides_src,
    int32_t dim,
    int32_t length
);

__global__ void expand_add(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_P0, int stride_P1
);

__global__ void expand_mul(
    float *Md, float *Nd, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1,
    int stride_P0, int stride_P1
);

__global__ void tensor_mul_kernel(
    float *dst, float *src1, float *src2,
    int32_t *shape,
    int32_t *strides_dst,
    int32_t *strides_src1,
    int32_t *strides_src2,
    int32_t dim,
    int32_t length
);

__global__ void tensor_add_kernel(
    float *dst, float *src1, float *src2,
    int32_t *shape,
    int32_t *strides_dst,
    int32_t *strides_src1,
    int32_t *strides_src2,
    int32_t dim,
    int32_t length
);

__global__ void tensor_sum_2d_dim0(
    float *Md, float *Pd,
    int M, int N,
    int stride_M0, int stride_M1
);

__global__ void cross_entropy(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *loss,
    int M, int N,
    int stride0, int stride1
);

__global__ void cross_entropy_backward(
    float *Md, int32_t *labels,
    float *maxs, float *sums,
    float *grad,
    int M, int N,
    int Md_stride0, int Md_stride1,
    int grad_stride0, int grad_stride1
);

__global__ void tensor_relu(
    float *Md, float *Nd, int M
);

__global__ void tensor_relu_prime(
    float *Md, float *Nd, int M
);

__global__ void tensor_l2_norm(
    float *Md, float *Nd, int M
);

__global__ void tensor_clip(
    float *Md, float *Norm, int M,
    float clip_value
);

__global__ void tensor_adam_step(
    float *w, float *grad,
    float *m, float *v,
    int M, int t,
    float beta1, float beta2,
    float lr, float eps
);

__global__ void reshape_deep_cp_float_kernel(
    float *dst, float *src,
    int32_t *src_shape, int32_t *src_strides,
    int32_t dim, int32_t length
);

__global__ void repeat_interleave_int32_kernel(
    int32_t *src, int32_t *dst,
    int32_t width,
    int32_t src_length, int32_t dst_length,
    int32_t n
);

__global__ void sequence_mask_kernel(
    float *src, int32_t *mask, float *dst,
    int M, int N,
    int l_stride0,
    int l_stride1,
    int m_stride0,
    int r_stride0,
    int r_stride1,
    float value
);

__global__ void softmax_kernel(
    float *src, float *dst,
    int shape0, int shape1, int shape2,
    int l_stride0, int l_stride1, int l_stride2,
    int r_stride0, int r_stride1, int r_stride2
);

__global__ void softmax_backward_kernel(
    float *target_grad, float *softmax_res, float *grad,
    int shape0, int shape1, int shape2,
    int t_stride0, int t_stride1, int t_stride2,
    int s_stride0, int s_stride1, int s_stride2,
    int g_stride0, int g_stride1, int g_stride2
);

__global__ void tensor_div_scalar(
    float *dst, float *src,
    int length, float value
);

__global__ void build_dropout_mask_kernel(
    float *mask,
    int32_t *shape,
    int32_t *strides,
    int length, int dim, float p
);

__global__ void tensor_embedding_kernel(
    float *dst,
    int32_t *indices,
    float *src,
    int src_shape0, int src_shape1,
    int length,
    int src_stride0, int src_strid1,
    int dst_stride0, int dst_stride1
);

__global__ void tensor_embedding_backward_kernel(
    float *dst,
    int32_t *indices,
    float *src,
    int src_shape0, int src_shape1,
    int length,
    int src_stride0, int src_strid1,
    int dst_stride0, int dst_stride1
);

__global__ void tensor_sum_2d_dim0_v1(
    float *src, float *sum,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int sum_stride0
);

__global__ void tensor_sum_2d_dim1(
    float *src, float *sum,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int sum_stride0
);

__global__ void tensor_var_2d_dim1(
    float *src, float *avg, float *sum,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int sum_stride0
);

__global__ void tensor_norm_kernel(
    float *src, float *avg, float * var, float *dst,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int dst_stride0, int dst_stride1
);

__global__ void tensor_norm_backward_kernel(
    float *src, float *norm, float * var, float *tgt,
    int src_shape0, int src_shape1,
    int src_stride0, int src_stride1,
    int norm_stride0, int norm_stride1,
    int tgt_stride0, int tgt_stride1
);

__global__ void tensor_mul_scalar(
    float *dst, float *src,
    int length, float value
);

#endif // GCC_ASAN

#endif // V5_TENSOR_BACKENDS_GPU_KERNEL_CUH