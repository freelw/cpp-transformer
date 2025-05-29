#include "cuda_ops.h"

#ifndef GCC_ASAN

#include "kernel.cuh"
#include <random>
#include <chrono>

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    curandStatus_t err_ = (err);                                               \
    if (err_ != CURAND_STATUS_SUCCESS) {                                       \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

CUDAOps::CUDAOps() {
    curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;
    const unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CHECK(curandSetGeneratorOrdering(gen, order));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
}

void CUDAOps::add(
    Tensor *lhs, const Tensor *rhs, Tensor *res,
    Tensor *l_shape, Tensor *l_strides,
    Tensor *r_striedes, Tensor *res_striedes
) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(l_shape != nullptr);
    assert(l_strides != nullptr);
    assert(r_striedes != nullptr);
    assert(res_striedes != nullptr);
    
    auto length = lhs->length();
    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    tensor_add_kernel<<<gridDim, blockDim>>>(
        (float *)res->get_data(),
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (int32_t *)l_shape->get_data(),
        (int32_t *)res_striedes->get_data(),
        (int32_t *)l_strides->get_data(),
        (int32_t *)r_striedes->get_data(),
        lhs->get_dim(),
        length
    );
}

void CUDAOps::addEq(
    Tensor *lhs, const Tensor *rhs,
    Tensor *l_shape,
    Tensor *l_strides, Tensor *r_striedes
) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    assert(lshape == rshape);
    int dim = lhs->get_dim();
    auto length = lhs->length();

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    tensor_add_eq_kernel<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (int32_t *)l_shape->get_data(),
        (int32_t *)l_strides->get_data(),
        (int32_t *)r_striedes->get_data(),
        dim,
        length
    );
}

void CUDAOps::expandAdd(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    auto shape = lhs->get_shape();
    assert(shape.size() == 2);
    assert(rhs->get_shape().size() == 1);   
    assert(rhs->get_shape()[0] == shape[1]);
    assert(shape == res->get_shape());

    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    expand_add<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lstrides[0],
        lstrides[1],
        res_strides[0],
        res_strides[1]
    );
}

void CUDAOps::expandMul(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    int size = lhs->size();
    auto shape = lhs->get_shape();
    assert(shape.size() == 2);
    assert(rhs->get_shape().size() == 1);   
    assert(rhs->get_shape()[0] == shape[1]);
    assert(shape == res->get_shape());

    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    expand_mul<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lstrides[0],
        lstrides[1],
        res_strides[0],
        res_strides[1]
    );
}

void CUDAOps::at(Tensor *lhs, const Tensor *rhs, Tensor *res) {
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    assert(lhs->get_dim() == 2);
    assert(rhs->get_dim() == 2);
    assert(res->get_dim() == 2);

    assert(lshape[1] == rshape[0]);
    assert(res_shape[0] == lshape[0]);
    assert(res_shape[1] == rshape[1]);

    const int M = lshape[0];
    const int N = lshape[1];
    const int P = rshape[1];

    const int stride_M0 = lstrides[0];
    const int stride_M1 = lstrides[1];
    const int stride_N0 = rstrides[0];
    const int stride_N1 = rstrides[1];
    const int stride_P0 = res_strides[0];
    const int stride_P1 = res_strides[1];

    this->memset((float *)res->get_data(), 0, res->size());

    dim3 gridDim(
        (P + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(
        TILE_WIDTH,
        TILE_WIDTH
    );

    tensor_at_2d<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (float *)res->get_data(),
        M,
        N,
        P,
        stride_M0,
        stride_M1,
        stride_N0,
        stride_N1,
        stride_P0,
        stride_P1
    );
}

void CUDAOps::embedding(Tensor *lhs, const Tensor *indices, const Tensor *res) {
    assert(lhs != nullptr);
    assert(indices != nullptr);
    assert(res != nullptr);

    assert(lhs->is_contiguous());
    assert(res->is_contiguous());
    assert(indices->is_contiguous());
    assert(!lhs->is_view());
    assert(!res->is_view());
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 2);
    assert(indices->get_dim() == 1);

    auto lshape = lhs->get_shape();
    auto rshape = res->get_shape();
    auto length = indices->length();

    assert(rshape[0] == length);
    assert(rshape[1] == lshape[1]);

    auto lstrides = lhs->get_strides();
    auto rstrides = res->get_strides();

    dim3 gridDim(
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    tensor_embedding_kernel<<<gridDim, blockDim>>>(
        (float *)res->get_data(),
        (int32_t *)indices->get_data(),
        (float *)lhs->get_data(),
        lshape[0],
        lshape[1],
        length,
        lstrides[0],
        lstrides[1],
        rstrides[0],
        rstrides[1]
    );
}

void CUDAOps::embeddingBackward(Tensor *lhs, const Tensor *indices, Tensor *res) {
    assert(lhs != nullptr);
    assert(indices != nullptr);
    assert(res != nullptr);

    assert(lhs->is_contiguous());
    assert(res->is_contiguous());
    assert(indices->is_contiguous());
    assert(!lhs->is_view());
    assert(!res->is_view());
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 2);
    assert(indices->get_dim() == 1);

    auto lshape = lhs->get_shape();
    auto rshape = res->get_shape();
    auto length = indices->length();

    assert(rshape[1] == lshape[1]);
    assert(lshape[0] == length);

    auto lstrides = lhs->get_strides(); // small grad
    auto rstrides = res->get_strides(); // emb big grad

    dim3 gridDim(
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    tensor_embedding_backward_kernel<<<gridDim, blockDim>>>(
        (float *)res->get_data(),
        (int32_t *)indices->get_data(),
        (float *)lhs->get_data(),
        lshape[0],
        lshape[1],
        length,
        lstrides[0],
        lstrides[1],
        rstrides[0],
        rstrides[1]
    );
}

void CUDAOps::mul(
    Tensor *lhs, const Tensor *rhs, Tensor *res,
    Tensor *l_shape, Tensor *l_strides,
    Tensor *r_striedes, Tensor *res_striedes
) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(l_shape != nullptr);
    assert(l_strides != nullptr);
    assert(r_striedes != nullptr);
    assert(res_striedes != nullptr);
    
    auto length = lhs->length();
    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    tensor_mul_kernel<<<gridDim, blockDim>>>(
        (float *)res->get_data(),
        (float *)lhs->get_data(),
        (float *)rhs->get_data(),
        (int32_t *)l_shape->get_data(),
        (int32_t *)res_striedes->get_data(),
        (int32_t *)l_strides->get_data(),
        (int32_t *)r_striedes->get_data(),
        lhs->get_dim(),
        length
    );
}

void CUDAOps::sum(Tensor *lhs, Tensor *res, int dim) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(dim >= 0 && dim < lhs->get_dim());

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(dim == 0);
    auto lstrides = lhs->get_strides();
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);

    dim3 gridDim(
        shape[1],
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(1, TILE_WIDTH);

    tensor_sum_2d_dim0_v1<<<gridDim, blockDim, TILE_WIDTH*sizeof(float)>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lhs->get_strides()[0],
        lhs->get_strides()[1],
        res->get_strides()[0]
    );
}

void CUDAOps::relu(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_relu<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        length
    );
}

void CUDAOps::reluPrime(Tensor *lhs, Tensor *res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_relu_prime<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        length
    );
}

void CUDAOps::crossEntropy(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(lhs != nullptr);
    assert(labels != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    assert(res != nullptr);

    assert(lhs->get_shape().size() == 2);
    assert(labels->get_shape().size() == 1);
    assert(maxs->get_shape().size() == 1);
    assert(sums->get_shape().size() == 1);
    assert(res->get_shape().size() == 1);
    assert(lhs->get_shape()[0] == labels->get_shape()[0]);
    assert(lhs->get_shape()[0] == maxs->get_shape()[0]);
    assert(lhs->get_shape()[0] == sums->get_shape()[0]);
    assert(res->get_shape()[0] == sums->get_shape()[0]);

    auto lstrides = lhs->get_strides();

    this->memset((float *)res->get_data(), 0, res->size());
    this->memset((float *)maxs->get_data(), 0, maxs->size());
    this->memset((float *)sums->get_data(), 0, sums->size());

    dim3 gridDim(
        (lhs->get_shape()[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);


    cross_entropy<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (int32_t *)labels->get_data(),
        (float *)maxs->get_data(),
        (float *)sums->get_data(),
        (float *)res->get_data(),
        lhs->get_shape()[0],
        lhs->get_shape()[1],
        lstrides[0],
        lstrides[1]
    );
}

void CUDAOps::crossEntropyBackward(Tensor *lhs, const Tensor *labels, Tensor *maxs, Tensor *sums, Tensor *res) {
    assert(lhs != nullptr);
    assert(labels != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    assert(res != nullptr);

    int batch_size = lhs->get_shape()[0];
    int size = lhs->get_shape()[1];
    float *data = static_cast<float*>(lhs->get_data());
    float *res_data = static_cast<float*>(res->get_data());
    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();
    assert(lstrides.size() == 2);
    assert(res_strides.size() == 2);

    dim3 gridDim(
        (batch_size + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    cross_entropy_backward<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (int32_t *)labels->get_data(),
        (float *)maxs->get_data(),
        (float *)sums->get_data(),
        (float *)res->get_data(),
        batch_size,
        size,
        lstrides[0],
        lstrides[1],
        res_strides[0],
        res_strides[1]
    );
}

void CUDAOps::calcAllGradNorm(const std::vector<Tensor*> &grads, Tensor *norm) {
    assert(norm != nullptr);
    assert(norm->length() == 1);
    this->memset((float *)norm->get_data(), 0, norm->size());
    for (auto &grad : grads) {
        assert(grad != nullptr);
        auto length = grad->length();
        dim3 gridDim(
            (length + TILE_WIDTH - 1) / TILE_WIDTH
        );
        dim3 blockDim(TILE_WIDTH);
        tensor_l2_norm<<<gridDim, blockDim, TILE_WIDTH*sizeof(float)>>>(
            (float *)grad->get_data(),
            (float *)norm->get_data(),
            length
        );
    }
}

void CUDAOps::clipGrad(Tensor *grad, const Tensor *norm, float grad_clip_val) {
    assert(grad != nullptr);
    assert(norm != nullptr);

    assert(norm->get_shape().size() == 1);

    auto length = grad->length();
    auto norm_length = norm->length();
    assert(norm_length == 1);

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_clip<<<gridDim, blockDim>>>(
        (float *)grad->get_data(),
        (float *)norm->get_data(),
        length,
        grad_clip_val
    );
}

void CUDAOps::adamStep(Tensor *w, Tensor *grad, Tensor *m, Tensor *v, int t, float lr, float beta1, float beta2, float epsilon) {
    assert(w != nullptr);
    assert(grad != nullptr);
    assert(m != nullptr);
    assert(v != nullptr);

    assert(!w->is_view());
    assert(!grad->is_view());
    assert(!m->is_view());
    assert(!v->is_view());

    assert(w->get_shape() == grad->get_shape());
    assert(w->get_shape() == m->get_shape());
    assert(w->get_shape() == v->get_shape());

    auto length = w->length();

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    tensor_adam_step<<<gridDim, blockDim>>>(
        (float *)w->get_data(),
        (float *)grad->get_data(),
        (float *)m->get_data(),
        (float *)v->get_data(),
        length,
        t,
        beta1,
        beta2,
        lr,
        epsilon
    );
}

void CUDAOps::init_weight_gauss(Tensor *tensor, float mean, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<float> distribution_w(0.0, sigma);
    auto size = tensor->size();
    float *data = static_cast<float*>(::malloc(size));
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w) + mean;
    }
    this->cp_to_device(tensor, (char *)data, size);
    ::free(data);
}

void CUDAOps::init_weight_uniform(Tensor *tensor, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::uniform_real_distribution<float> distribution_w(-sigma, sigma);
    auto size = tensor->size();
    float *data = static_cast<float*>(::malloc(size));
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w);
    }
    this->cp_to_device(tensor, (char *)data, size);
    ::free(data);
}

void CUDAOps::init_weight_for_dbg(Tensor *tensor, float scale) {
    auto size = tensor->size();
    void *_data = ::malloc(size);

    if (tensor->get_dtype() == FLOAT32) {
        float *data = static_cast<float*>(_data);
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = static_cast<float>(i) * 1e-5 * scale;
        }
    } else if (tensor->get_dtype() == INT32) {
        int32_t *data = static_cast<int32_t*>(_data);
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = i % 10;
        }
    } else {
        assert(false);
    }
    this->cp_to_device(tensor, (char *)_data, size);
    ::free(_data);
}

void CUDAOps::fill(Tensor *tensor, float value) {
    dim3 gridDim(
        (tensor->length() + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);
    fill_float<<<gridDim, blockDim>>>(
        (float *)tensor->get_data(),
        tensor->length(),
        value
    );
}

void CUDAOps::reshape_deep_cp(
    Tensor *dst_tensor, const Tensor *src_tensor,
    const Tensor *src_shape, const Tensor *src_strides
) {
    assert(dst_tensor->get_dtype() == src_tensor->get_dtype());
    assert(
        dst_tensor->get_dtype() == INT32 ||
        dst_tensor->get_dtype() == FLOAT32
    );

    auto dtype = dst_tensor->get_dtype();
    auto src_shape_data = static_cast<int32_t*>(src_shape->get_data());
    auto src_strides_data = static_cast<int32_t*>(src_strides->get_data());
    auto dim = src_tensor->get_dim();
    auto length = src_tensor->length();

    if (dtype == INT32) {
        assert(false);
    } else if (dtype == FLOAT32) {
        dim3 gridDim(
            (length + TILE_WIDTH - 1) / TILE_WIDTH
        );
        dim3 blockDim(TILE_WIDTH);
        reshape_deep_cp_float_kernel<<<gridDim, blockDim>>>(
            (float *)dst_tensor->get_data(),
            (float *)src_tensor->get_data(),
            src_shape_data,
            src_strides_data,
            dim,
            length
        );
    } else {
        assert(false);
    }
}

void CUDAOps::repeat_interleave(Tensor *lhs, Tensor *res, int n) {
    assert(lhs->get_dtype() == INT32);
    assert(res->get_dtype() == INT32);
    assert(lhs != nullptr);
    assert(res != nullptr);

    auto lshape = lhs->get_shape();
    auto dim = lhs->get_dim();
    assert(dim > 0);
    int width = 0;
    
    if (dim == 1) {
        width = 1;
    } else {
        width = lshape[dim-1];   
    }
    auto l_length = lhs->length();
    auto r_length = res->length();
    assert(l_length * n == r_length);
    assert(l_length % width == 0);

    dim3 gridDim(
        (r_length + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH);

    repeat_interleave_int32_kernel<<<gridDim, blockDim>>>(
        (int32_t *)lhs->get_data(),
        (int32_t *)res->get_data(),
        width,
        l_length,
        r_length,
        n
    );
}

void CUDAOps::sequence_mask(Tensor *lhs, const Tensor *mask, Tensor *res, float value) {
    assert(lhs != nullptr);
    assert(mask != nullptr);
    assert(res != nullptr);

    assert(lhs->get_dim() == 2);
    assert(mask->get_dim() == 1);
    assert(res->get_dim() == 2);

    auto lshape = lhs->get_shape();
    auto mshape = mask->get_shape();
    auto rshape = res->get_shape();

    assert(lshape[0] == mshape[0]);
    assert(lshape[1] == rshape[1]);
    assert(rshape[0] == mshape[0]);

    auto lstrides = lhs->get_strides();
    auto mstrides = mask->get_strides();
    auto rstrides = res->get_strides();

    dim3 gridDim(
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    sequence_mask_kernel<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (int32_t *)mask->get_data(),
        (float *)res->get_data(),
        lshape[0],
        lshape[1],
        lstrides[0],
        lstrides[1],
        mstrides[0],
        rstrides[0],
        rstrides[1],
        value
    );
}

void CUDAOps::softmax(Tensor *lhs, Tensor *res) {
    auto l_shape = lhs->get_shape();
    auto r_shape = res->get_shape();
    assert(l_shape == r_shape);
    assert(lhs->get_dtype() == FLOAT32);
    assert(res->get_dtype() == FLOAT32);
    assert(lhs->get_dim() == 3);
    assert(res->get_dim() == 3);
    auto lstrides = lhs->get_strides();
    auto rstrides = res->get_strides();

    dim3 gridDim(
        (l_shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (l_shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    softmax_kernel<<<gridDim, blockDim>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        l_shape[0],
        l_shape[1],
        l_shape[2],
        lstrides[0],
        lstrides[1],
        lstrides[2],
        rstrides[0],
        rstrides[1],
        rstrides[2]
    );
}

void CUDAOps::softmax_bacward(Tensor *target_grad, const Tensor *softmax_res, Tensor *grad) {
    assert(target_grad != nullptr);
    assert(softmax_res != nullptr);
    assert(grad != nullptr);

    assert(target_grad->get_dtype() == FLOAT32);
    assert(softmax_res->get_dtype() == FLOAT32);
    assert(grad->get_dtype() == FLOAT32);

    assert(target_grad->get_dim() == 3);
    assert(softmax_res->get_dim() == 3);
    assert(grad->get_dim() == 3);

    auto t_shape = target_grad->get_shape();
    auto s_shape = softmax_res->get_shape();
    auto g_shape = grad->get_shape();

    assert(t_shape == s_shape);
    assert(t_shape == g_shape);

    auto t_strides = target_grad->get_strides();
    auto s_strides = softmax_res->get_strides();
    auto g_strides = grad->get_strides();

    dim3 gridDim(
        (t_shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (t_shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    softmax_backward_kernel<<<gridDim, blockDim>>>(
        (float *)target_grad->get_data(),
        (float *)softmax_res->get_data(),
        (float *)grad->get_data(),
        t_shape[0],
        t_shape[1],
        t_shape[2],
        t_strides[0],
        t_strides[1],
        t_strides[2],
        s_strides[0],
        s_strides[1],
        s_strides[2],
        g_strides[0],
        g_strides[1],
        g_strides[2]
    );
}

void CUDAOps::div(Tensor *dst, Tensor *src, float value) {
    assert(dst->length() == src->length());
    assert(dst->get_shape() == src->get_shape());
    assert(dst->get_strides() == src->get_strides());
    auto length = dst->length();
    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);
    tensor_div_scalar<<<gridDim, blockDim>>>(
        (float *)dst->get_data(),
        (float *)src->get_data(),
        length,
        value
    );
}

void CUDAOps::build_dropout_mask(
    Tensor *mask, float p,
    Tensor *shape, Tensor *strides
) {
    assert(mask != nullptr);
    CURAND_CHECK(curandGenerateUniform(
        gen,
        reinterpret_cast<float*>(mask->get_data()),
        mask->length())
    );
    dim3 gridDim(
        (mask->length() + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);
    build_dropout_mask_kernel<<<gridDim, blockDim>>>(
        (float *)mask->get_data(),
        (int32_t *)shape->get_data(),
        (int32_t *)strides->get_data(),
        mask->length(),
        mask->get_dim(),
        p
    );
}

void CUDAOps::pos_encoding(Tensor *res) {
    assert(res != nullptr);
    auto shape = res->get_shape();
    auto max_len = shape[0];
    auto num_hidden = shape[1];
    float *data = static_cast<float*>(::malloc(res->size()));
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < num_hidden; ++i) {
            if (i % 2 == 0) {
                data[pos * res->get_strides()[0] + i * res->get_strides()[1]] = 
                    std::sin(pos * 1. / std::pow(10000, (1.0f * i / num_hidden)));
            } else {
                data[pos * res->get_strides()[0] + i * res->get_strides()[1]] = 
                    std::cos(pos * 1. / std::pow(10000, (1.0f * (i & ~1) / num_hidden)));
            }
        }
    }
    this->cp_to_device(res, (char *)data, res->size());
    ::free(data);
}

void CUDAOps::avg(Tensor *lhs, Tensor *res) {
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);
    auto shape = lhs->get_shape();
    assert(shape[0] == res->get_shape()[0]);

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        shape[0]
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_sum_2d_dim1<<<gridDim, blockDim, TILE_WIDTH*sizeof(float)>>>(
        (float *)lhs->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lhs->get_strides()[0],
        lhs->get_strides()[1],
        res->get_strides()[0]
    );

    auto length = res->length();
    dim3 gridDim_div(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim_div(TILE_WIDTH);

    tensor_div_scalar<<<gridDim_div, blockDim_div>>>(
        (float *)res->get_data(),
        (float *)res->get_data(),
        length,
        shape[1]
    );
}

void CUDAOps::var(Tensor *lhs, const Tensor *_avg, Tensor *res) {
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);
    assert(_avg->get_dim() == 1);
    auto shape = lhs->get_shape();
    assert(shape[0] == res->get_shape()[0]);
    assert(shape[0] == _avg->get_shape()[0]);

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        shape[0]
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_var_2d_dim1<<<gridDim, blockDim, TILE_WIDTH*sizeof(float)>>>(
        (float *)lhs->get_data(),
        (float *)_avg->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        lhs->get_strides()[0],
        lhs->get_strides()[1],
        res->get_strides()[0]
    );

    auto length = res->length();
    dim3 gridDim_div(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim_div(TILE_WIDTH);

    tensor_div_scalar<<<gridDim_div, blockDim_div>>>(
        (float *)res->get_data(),
        (float *)res->get_data(),
        length,
        shape[1]
    );
}

void CUDAOps::norm(const Tensor *src, const Tensor *avg, const Tensor *var, Tensor *res) {
    assert(src->get_dim() == 2);
    assert(avg->get_dim() == 1);
    assert(var->get_dim() == 1);
    assert(res->get_dim() == 2);
    assert(src->get_shape() == res->get_shape());
    auto shape = src->get_shape();
    assert(shape[0] == avg->get_shape()[0]);
    assert(shape[0] == var->get_shape()[0]);
    auto src_strides = src->get_strides();
    auto res_strides = res->get_strides();

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    tensor_norm_kernel<<<gridDim, blockDim>>>(
        (float *)src->get_data(),
        (float *)avg->get_data(),
        (float *)var->get_data(),
        (float *)res->get_data(),
        shape[0],
        shape[1],
        src_strides[0],
        src_strides[1],
        res_strides[0],
        res_strides[1]
    );
}

void CUDAOps::normBackward(
    const Tensor *src_grad, const Tensor *norm_res, const Tensor *var_res, Tensor *tgt_grad
)  {
    assert(src_grad != nullptr);
    assert(norm_res != nullptr);
    assert(tgt_grad != nullptr);
    assert(src_grad->get_dim() == 2);
    assert(norm_res->get_dim() == 2);
    assert(tgt_grad->get_dim() == 2);
    assert(src_grad->get_shape() == tgt_grad->get_shape());
    assert(src_grad->get_shape() == norm_res->get_shape());
    assert(var_res->get_dim() == 1);
    auto shape = src_grad->get_shape();
    assert(shape[0] == var_res->get_shape()[0]);
    auto norm_res_strides = norm_res->get_strides();
    auto src_grad_strides = src_grad->get_strides();
    auto tgt_grad_strides = tgt_grad->get_strides();

    dim3 gridDim(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH
    );

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    tensor_norm_backward_kernel<<<gridDim, blockDim>>>(
        (float *)src_grad->get_data(),
        (float *)norm_res->get_data(),
        (float *)var_res->get_data(),
        (float *)tgt_grad->get_data(),
        shape[0],
        shape[1],
        src_grad_strides[0],
        src_grad_strides[1],
        norm_res_strides[0],
        norm_res_strides[1],
        tgt_grad_strides[0],
        tgt_grad_strides[1]
    );
}

void CUDAOps::mulSV(Tensor *dst, Tensor *src, float value) {
    assert(dst->length() == src->length());
    assert(dst->get_shape() == src->get_shape());
    assert(dst->get_strides() == src->get_strides());
    auto length = dst->length();

    dim3 gridDim(
        (length + TILE_WIDTH - 1) / TILE_WIDTH
    );
    dim3 blockDim(TILE_WIDTH);

    tensor_mul_scalar<<<gridDim, blockDim>>>(
        (float *)dst->get_data(),
        (float *)src->get_data(),
        length,
        value
    );
}

void* CUDAOps::alloc(size_t size) {
    void *ret = nullptr;
    cudaMalloc((void **)&ret, size);
    return ret;
}

void CUDAOps::memset(void* ptr, int value, size_t size) {
    ::cudaMemset(ptr, value, size);
}

void CUDAOps::cp_device_to_device(void* dst, const void* src, size_t size) {
    ::cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void CUDAOps::free(void* ptr) {
    ::cudaFree(ptr);
}

void CUDAOps::cp_to_device(Tensor *dst_tensor, char *src, size_t size) {
    ::cudaMemcpy(dst_tensor->get_data(), src, size, cudaMemcpyHostToDevice);
}

void CUDAOps::cp_from_device(char *dst, const Tensor *src_tensor, size_t size) {
    ::cudaMemcpy(dst, src_tensor->get_data(), size, cudaMemcpyDeviceToHost);
}
#endif // GCC_ASAN