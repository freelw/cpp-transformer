#include "cpu_ops.h"
#include <string.h>
#include <cmath>
#include <random>
#include <chrono>

CPUOps::CPUOps() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = std::mt19937(seed);
    dis = std::uniform_real_distribution<>(0, 1);
}

void CPUOps::add(
    Tensor* lhs, const Tensor* rhs, Tensor* res,
    Tensor*/*l_shape*/, Tensor*/*l_strides*/,
    Tensor*/*r_striedes*/,
    Tensor*/*res_striedes*/
) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    for (int i = 0; i < lhs->length(); ++i) {
        int index_l = 0;
        int index_r = 0;
        int index_res = 0;
        int tmp_index = i;
        int tot_length = lhs->length();
        for (int j = 0; j < lhs->get_dim(); ++j) {
            tot_length /= lhs->get_shape()[j];
            int cur_dim_index = tmp_index / tot_length;
            index_l += cur_dim_index * lstrides[j];
            index_r += cur_dim_index * rstrides[j];
            index_res += cur_dim_index * res_strides[j];
            tmp_index %= tot_length;
        }
        static_cast<float*>(res->get_data())[index_res] =
            static_cast<float*>(lhs->get_data())[index_l] +
            static_cast<float*>(rhs->get_data())[index_r];
    }
}

void CPUOps::addEq(
    Tensor* lhs, const Tensor* rhs,
    Tensor*/*l_shape*/,
    Tensor*/*l_strides*/, Tensor*/*r_striedes*/
) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    assert(lshape == rshape);
    for (int i = 0; i < lhs->length(); ++i) {
        int index_l = 0;
        int index_r = 0;
        int tmp_index = i;
        int tot_length = lhs->length();
        for (int j = 0; j < lhs->get_dim(); ++j) {
            tot_length /= lhs->get_shape()[j];
            int cur_dim_index = tmp_index / tot_length;
            index_l += cur_dim_index * lhs->get_strides()[j];
            index_r += cur_dim_index * rhs->get_strides()[j];
            tmp_index %= tot_length;
        }
        static_cast<float*>(lhs->get_data())[index_l] +=
            static_cast<float*>(rhs->get_data())[index_r];
    }
}

void CPUOps::expandAdd(Tensor* lhs, const Tensor* rhs, Tensor* res) {
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

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * res_strides[0] + j * res_strides[1]] =
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]] +
                static_cast<float*>(rhs->get_data())[j];
        }
    }
}

void CPUOps::expandMul(Tensor* lhs, const Tensor* rhs, Tensor* res) {
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

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * res_strides[0] + j * res_strides[1]] =
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]] *
                static_cast<float*>(rhs->get_data())[j];
        }
    }
}

void CPUOps::at(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();
    assert(lshape.size() == 2);
    assert(rshape.size() == 2);
    assert(res_shape.size() == 2);
    assert(lshape[1] == rshape[0]);
    assert(res_shape[0] == lshape[0]);
    assert(res_shape[1] == rshape[1]);

    float* res_data = static_cast<float*>(res->get_data());
    float* lhs_data = static_cast<float*>(lhs->get_data());
    float* rhs_data = static_cast<float*>(rhs->get_data());

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    for (int i = 0; i < lshape[0]; ++i) {
        for (int j = 0; j < rshape[1]; ++j) {
            res_data[i * res_strides[0] + j * res_strides[1]] = 0;
            for (int k = 0; k < lshape[1]; ++k) {
                res_data[i * res_strides[0] + j * res_strides[1]] +=
                    lhs_data[i * lstrides[0] + k * lstrides[1]] *
                    rhs_data[k * rstrides[0] + j * rstrides[1]];
            }
        }
    }
}

void CPUOps::embedding(Tensor* lhs, const Tensor* indices, const Tensor* res) {
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

    for (int i = 0; i < length; ++i) {
        int index = static_cast<int32_t*>(indices->get_data())[i];
        assert(index >= 0 && index < lshape[0]);
        for (int j = 0; j < lshape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * res->get_strides()[0] + j * res->get_strides()[1]] =
                static_cast<float*>(lhs->get_data())[index * lhs->get_strides()[0] + j * lhs->get_strides()[1]];
        }
    }
}

void CPUOps::embeddingBackward(Tensor* lhs, const Tensor* indices, Tensor* res) {
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

    for (int i = 0; i < length; ++i) {
        int index = static_cast<int32_t*>(indices->get_data())[i];
        assert(index >= 0 && index < rshape[0]);
        for (int j = 0; j < lshape[1]; ++j) {
            static_cast<float*>(res->get_data())[index * res->get_strides()[0] + j * res->get_strides()[1]] +=
                static_cast<float*>(lhs->get_data())[i * lhs->get_strides()[0] + j * lhs->get_strides()[1]];
        }
    }
}

void CPUOps::mul(
    Tensor* lhs, const Tensor* rhs, Tensor* res,
    Tensor*/*l_shape*/, Tensor*/*l_strides*/,
    Tensor*/*r_striedes*/,
    Tensor*/*res_striedes*/
) {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);

    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    auto res_shape = res->get_shape();

    auto lstrides = lhs->get_strides();
    auto rstrides = rhs->get_strides();
    auto res_strides = res->get_strides();

    for (int i = 0; i < lhs->length(); ++i) {
        int index_l = 0;
        int index_r = 0;
        int index_res = 0;
        int tmp_index = i;
        int tot_length = lhs->length();
        for (int j = 0; j < lhs->get_dim(); ++j) {
            tot_length /= lhs->get_shape()[j];
            int cur_dim_index = tmp_index / tot_length;
            index_l += cur_dim_index * lstrides[j];
            index_r += cur_dim_index * rstrides[j];
            index_res += cur_dim_index * res_strides[j];
            tmp_index %= tot_length;
        }
        static_cast<float*>(res->get_data())[index_res] =
            static_cast<float*>(lhs->get_data())[index_l] *
            static_cast<float*>(rhs->get_data())[index_r];
    }
}

void CPUOps::sum(Tensor* lhs, Tensor* res, int dim) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(dim >= 0 && dim < lhs->get_dim());

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(dim == 0);

    auto lstrides = lhs->get_strides();
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);

    for (int i = 0; i < shape[1]; ++i) {
        static_cast<float*>(res->get_data())[i] = 0;
        float tmp = 0;
        for (int j = 0; j < shape[0]; ++j) {
            tmp += static_cast<float*>(lhs->get_data())[j * lstrides[0] + i * lstrides[1]];
        }
        static_cast<float*>(res->get_data())[i] = tmp;
    }
}

void CPUOps::relu(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(res->get_data())[i] =
            std::max(0.0f, static_cast<float*>(lhs->get_data())[i]);
    }
}

void CPUOps::reluPrime(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(res->get_data())[i] =
            static_cast<float*>(lhs->get_data())[i] > 0 ? 1.0f : 0.0f;
    }
}

void CPUOps::crossEntropy(Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res) {
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

    int batch_size = lhs->get_shape()[0];
    int size = lhs->get_shape()[1];
    float loss_value = 0;

    // maxs and sums are out params
    float* data = static_cast<float*>(lhs->get_data());
    int32_t* labels_data = static_cast<int32_t*>(labels->get_data());
    auto lstrides = lhs->get_strides();
    for (int j = 0; j < batch_size; ++j) {
        float max = data[j * lstrides[0]];
        for (int i = 0; i < size; ++i) {
            auto e = data[j * lstrides[0] + i * lstrides[1]];
            if (max < e) {
                max = e;
            }
        }
        float sum = 0;
        auto target = labels_data[j];
        assert(target >= 0 && target < size);
        float zt = data[j * lstrides[0] + target * lstrides[1]];
        for (int i = 0; i < size; ++i) {
            float e = data[j * lstrides[0] + i * lstrides[1]];
            e = std::exp(e - max);
            sum += e;
        }
        static_cast<float*>(maxs->get_data())[j] = max;
        static_cast<float*>(sums->get_data())[j] = sum;
        static_cast<float*>(res->get_data())[j] = -(zt - max - std::log(sum));

        if (std::isnan(static_cast<float*>(res->get_data())[j])) {
            std::cerr << "CrossEntropy loss is NaN at batch " << j << ", max: " << max
                << ", sum: " << sum << ", zt: " << zt << std::endl;
            std::cerr << "lstrides[0] = " << lstrides[0] << ", lstrides[1] = " << lstrides[1] << std::endl;
            for (int i = 0; i < size; ++i) {
                auto e = data[j * lstrides[0] + i * lstrides[1]];
                std::cerr << "data[" << j << "][" << i << "] = " << e << std::endl;
            }

            validateAllTensors();
            abort();
        }
    }
}

void CPUOps::crossEntropyBackward(Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res) {
    assert(lhs != nullptr);
    assert(labels != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    assert(res != nullptr);

    int batch_size = lhs->get_shape()[0];
    int size = lhs->get_shape()[1];
    float* data = static_cast<float*>(lhs->get_data());
    float* res_data = static_cast<float*>(res->get_data());
    auto lstrides = lhs->get_strides();
    auto res_strides = res->get_strides();
    assert(lstrides.size() == 2);
    assert(res_strides.size() == 2);

    for (int j = 0; j < batch_size; ++j) {
        float max = static_cast<float*>(maxs->get_data())[j];
        float sum = static_cast<float*>(sums->get_data())[j];
        auto target = static_cast<int32_t*>(labels->get_data())[j];
        for (int i = 0; i < size; ++i) {
            if (i == target) {
                res_data[j * res_strides[0] + i * res_strides[1]] =
                    (std::exp(data[j * lstrides[0] + i * lstrides[1]] - max) / sum - 1);
            } else {
                res_data[j * res_strides[0] + i * res_strides[1]] =
                    (std::exp(data[j * lstrides[0] + i * lstrides[1]] - max) / sum);
            }
        }
    }
}


void CPUOps::calcAllGradNorm(const std::vector<Tensor*>& grads, Tensor* norm) {
    float tmp = 0;
    for (auto grad : grads) {
        float* data = static_cast<float*>(grad->get_data());
        for (int i = 0; i < grad->length(); ++i) {
            float& value = data[i];
            tmp += std::pow(value, 2);
        }
    }
    assert(norm->get_shape().size() == 1);
    assert(norm->get_shape()[0] == 1);
    float* norm_data = static_cast<float*>(norm->get_data());
    norm_data[0] = tmp;
}

void CPUOps::clipGrad(Tensor* grad, const Tensor* norm, float grad_clip_val) {
    assert(grad != nullptr);
    assert(norm != nullptr);
    assert(norm->get_shape().size() == 1);
    assert(norm->get_shape()[0] == 1);
    float* data = static_cast<float*>(grad->get_data());
    float* norm_data = static_cast<float*>(norm->get_data());
    float norm_value = std::sqrt(norm_data[0]);
    if (norm_value > grad_clip_val) {
        for (int i = 0; i < grad->length(); ++i) {
            data[i] *= grad_clip_val / norm_value;
        }
    }
}

void CPUOps::adamStep(Tensor* w, Tensor* grad, Tensor* m, Tensor* v, int t, float lr, float beta1, float beta2, float epsilon) {
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

    float* w_data = static_cast<float*>(w->get_data());
    float* grad_data = static_cast<float*>(grad->get_data());
    float* m_data = static_cast<float*>(m->get_data());
    float* v_data = static_cast<float*>(v->get_data());
    for (int i = 0; i < w->length(); ++i) {
        m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
        v_data[i] = beta2 * v_data[i] + (1 - beta2) * std::pow(grad_data[i], 2);
        float m_hat = m_data[i] / (1 - std::pow(beta1, t));
        float v_hat = v_data[i] / (1 - std::pow(beta2, t));
        w_data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

void CPUOps::init_weight_gauss(Tensor* tensor, float mean, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<float> distribution_w(0.0, sigma);
    float* data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w) + mean;
    }
}

void CPUOps::init_weight_uniform(Tensor* tensor, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::uniform_real_distribution<float> distribution_w(-sigma, sigma);
    float* data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w);
    }
}

void CPUOps::init_weight_for_dbg(Tensor* tensor, float scale) {
    assert(tensor != nullptr);
    assert(tensor->get_data() != nullptr);
    assert(tensor->length() > 0);

    if (tensor->get_dtype() == FLOAT32) {
        float* data = static_cast<float*>(tensor->get_data());
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = static_cast<float>(i) * 1e-5 * scale;
        }
    } else if (tensor->get_dtype() == INT32) {
        int32_t* data = static_cast<int32_t*>(tensor->get_data());
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = i % 10;
        }
    } else {
        assert(false);
    }

}

void CPUOps::fill(Tensor* tensor, float value) {
    assert(tensor != nullptr);
    assert(tensor->get_data() != nullptr);
    assert(tensor->length() > 0);

    float* data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = value;
    }
}

void CPUOps::reshape_deep_cp(
    Tensor* dst_tensor, const Tensor* src_tensor,
    const Tensor* src_shape, const Tensor* src_strides) {

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
        auto dst_data = static_cast<float*>(dst_tensor->get_data());
        auto src_data = static_cast<float*>(src_tensor->get_data());
        for (int i = 0; i < length; ++i) {
            int offset = 0;
            int index = i;
            auto tmp_length = length;
            for (int j = 0; j < dim; ++j) {
                tmp_length /= src_shape_data[j];
                auto cur_dim_index = index / tmp_length;
                offset += cur_dim_index * src_strides_data[j];
                index %= tmp_length;
            }
            dst_data[i] = src_data[offset];
        }
    } else {
        assert(false);
    }
}

void CPUOps::repeat_interleave(Tensor* lhs, Tensor* res, int n) {
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
        width = lshape[dim - 1];
    }
    auto l_length = lhs->length();
    auto r_length = res->length();
    assert(l_length * n == r_length);
    assert(l_length % width == 0);
    auto blocks = l_length / width;

    for (int i = 0; i < blocks; ++i) {
        int src_offset = i * width;
        int tgt_offset = i * width * n;
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < width; ++k) {
                static_cast<int32_t*>(res->get_data())[tgt_offset + k] =
                    static_cast<int32_t*>(lhs->get_data())[src_offset + k];
            }
            tgt_offset += width;
        }
    }
}

void CPUOps::sequence_mask(Tensor* lhs, const Tensor* mask, Tensor* res, float value) {
    assert(lhs != nullptr);
    assert(mask != nullptr);
    assert(res != nullptr);

    assert(lhs->get_dtype() == FLOAT32);
    assert(mask->get_dtype() == INT32);
    assert(res->get_dtype() == FLOAT32);

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

    for (int i = 0; i < lshape[0]; ++i) {
        for (int j = 0; j < lshape[1]; ++j) {
            static_cast<float*>(res->get_data())[i * rstrides[0] + j * rstrides[1]] =
                static_cast<int32_t*>(mask->get_data())[i * mstrides[0]] <= j ? value :
                static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]];
        }
    }
}

void CPUOps::softmax(Tensor* lhs, Tensor* res) {
    auto l_shape = lhs->get_shape();
    auto r_shape = res->get_shape();
    assert(l_shape == r_shape);
    assert(lhs->get_dtype() == FLOAT32);
    assert(res->get_dtype() == FLOAT32);
    assert(lhs->get_dim() == 3);
    assert(res->get_dim() == 3);
    auto lstrides = lhs->get_strides();
    auto rstrides = res->get_strides();
    for (int i = 0; i < l_shape[0]; ++i) {
        for (int j = 0; j < l_shape[1]; ++j) {
            float max = static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1]];
            for (int k = 0; k < l_shape[2]; ++k) {
                auto e = static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]];
                if (max < e) {
                    max = e;
                }
            }
            float sum = 0;
            for (int k = 0; k < l_shape[2]; ++k) {
                float e = static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]];
                e = std::exp(e - max);
                sum += e;
            }
            for (int k = 0; k < l_shape[2]; ++k) {
                static_cast<float*>(res->get_data())[i * rstrides[0] + j * rstrides[1] + k * rstrides[2]] =
                    std::exp(static_cast<float*>(lhs->get_data())[i * lstrides[0] + j * lstrides[1] + k * lstrides[2]] - max) / sum;
            }
        }
    }
}

void CPUOps::softmax_bacward(Tensor* target_grad, const Tensor* softmax_res, Tensor* grad) {
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

    float* target_grad_data = static_cast<float*>(target_grad->get_data());
    float* softmax_res_data = static_cast<float*>(softmax_res->get_data());
    float* grad_data = static_cast<float*>(grad->get_data());

    for (int i = 0; i < t_shape[0]; ++i) {
        for (int j = 0; j < t_shape[1]; ++j) {
            for (int target = 0; target < t_shape[2]; ++target) {
                auto tg_target_pos = i * t_strides[0] + j * t_strides[1] + target * t_strides[2];
                float tmp = 0;
                for (int k = 0; k < t_shape[2]; ++k) {
                    // auto tg_k_pos = i * t_strides[0] + j * t_strides[1] + k * t_strides[2];
                    auto sm_target_pos = i * s_strides[0] + j * s_strides[1] + target * s_strides[2];
                    auto sm_k_pos = i * s_strides[0] + j * s_strides[1] + k * s_strides[2];
                    // auto g_target_pos = i * g_strides[0] + j * g_strides[1] + target * g_strides[2];
                    auto g_k_pos = i * g_strides[0] + j * g_strides[1] + k * g_strides[2];
                    auto softmax_res_k = softmax_res_data[sm_k_pos];
                    auto softmax_res_target = softmax_res_data[sm_target_pos];
                    auto grad_k = grad_data[g_k_pos];
                    if (target == k) {
                        tmp += softmax_res_k * (1 - softmax_res_k) * grad_k;
                    } else {
                        tmp += -softmax_res_target * softmax_res_k * grad_k;
                    }
                }
                target_grad_data[tg_target_pos] = tmp;
            }
        }
    }
}

void CPUOps::div(Tensor* dst, Tensor* src, float value) {
    assert(dst->length() == src->length());
    auto length = dst->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(dst->get_data())[i] =
            static_cast<float*>(src->get_data())[i] / value;
    }
}

void CPUOps::build_dropout_mask(
    Tensor* mask, float p,
    Tensor*/*shape*/, Tensor*/*strides*/
) {
    assert(mask != nullptr);
    // assert(mask->get_dim() == 1);
    auto length = mask->length();
    for (int i = 0; i < length; ++i) {
        int index = 0;
        int tmp_index = i;
        int tot_length = length;
        for (int j = 0; j < mask->get_dim(); ++j) {
            tot_length /= mask->get_shape()[j];
            int l = tmp_index / tot_length;
            index += l * mask->get_strides()[j];
            tmp_index %= tot_length;
        }
        static_cast<float*>(mask->get_data())[i] = dis(gen) < p ? 0.0f : 1.0f;
    }
}

void CPUOps::pos_encoding(Tensor* res) {
    assert(res != nullptr);
    auto shape = res->get_shape();
    auto max_len = shape[0];
    auto num_hidden = shape[1];
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < num_hidden; ++i) {
            if (i % 2 == 0) {
                static_cast<float*>(res->get_data())[pos * res->get_strides()[0] + i * res->get_strides()[1]] =
                    std::sin(pos * 1. / std::pow(10000, (1.0f * i / num_hidden)));
            } else {
                static_cast<float*>(res->get_data())[pos * res->get_strides()[0] + i * res->get_strides()[1]] =
                    std::cos(pos * 1. / std::pow(10000, (1.0f * (i & ~1) / num_hidden)));
            }
        }
    }
}

void CPUOps::avg(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);
    assert(lhs->get_shape()[0] == res->get_shape()[0]);

    auto shape = lhs->get_shape();
    for (int i = 0; i < shape[0]; ++i) {
        float sum = 0;
        for (int j = 0; j < shape[1]; ++j) {
            sum += static_cast<float*>(lhs->get_data())[i * lhs->get_strides()[0] + j * lhs->get_strides()[1]];
        }
        static_cast<float*>(res->get_data())[i] = sum / shape[1];
    }
}

void CPUOps::var(Tensor* lhs, const Tensor* _avg, Tensor* res) {
    assert(lhs != nullptr);
    assert(_avg != nullptr);
    assert(res != nullptr);
    assert(lhs->get_dim() == 2);
    assert(_avg->get_dim() == 1);
    assert(res->get_dim() == 1);
    assert(lhs->get_shape()[0] == res->get_shape()[0]);
    assert(lhs->get_shape()[0] == _avg->get_shape()[0]);

    auto shape = lhs->get_shape();
    float* avg = static_cast<float*>(_avg->get_data());
    for (int i = 0; i < shape[0]; ++i) {
        float sum = 0;
        for (int j = 0; j < shape[1]; ++j) {
            float v = static_cast<float*>(lhs->get_data())[i * lhs->get_strides()[0] + j * lhs->get_strides()[1]];
            float diff = v - avg[i];
            sum += std::pow(diff, 2);
        }
        static_cast<float*>(res->get_data())[i] = sum / shape[1];
    }
}

void CPUOps::norm(const Tensor* src, const Tensor* avg, const Tensor* var, Tensor* res) {
    assert(src->get_dim() == 2);
    assert(avg->get_dim() == 1);
    assert(var->get_dim() == 1);
    assert(res->get_dim() == 2);
    assert(src->get_shape() == res->get_shape());
    const float eps = 1e-5;
    auto shape = src->get_shape();
    assert(shape[0] == avg->get_shape()[0]);
    assert(shape[0] == var->get_shape()[0]);
    auto src_strides = src->get_strides();
    auto res_strides = res->get_strides();

    for (int i = 0; i < shape[0]; ++i) {
        float avg_value = static_cast<float*>(avg->get_data())[i];
        float var_value = static_cast<float*>(var->get_data())[i];
        for (int j = 0; j < shape[1]; ++j) {
            float v = static_cast<float*>(src->get_data())[i * src_strides[0] + j * src_strides[1]];
            static_cast<float*>(res->get_data())[i * res_strides[0] + j * res_strides[1]] =
                (v - avg_value) / std::sqrt(var_value + eps);
        }
    }
}

void CPUOps::normBackward(
    const Tensor* src_grad, const Tensor* norm_res, const Tensor* var_res, Tensor* tgt_grad
) {
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
    const float eps = 1e-5;
    auto norm_res_strides = norm_res->get_strides();
    auto src_grad_strides = src_grad->get_strides();
    auto tgt_grad_strides = tgt_grad->get_strides();
    float* norm_res_data = static_cast<float*>(norm_res->get_data());
    float* src_grad_data = static_cast<float*>(src_grad->get_data());
    float* tgt_grad_data = static_cast<float*>(tgt_grad->get_data());

    for (int k = 0; k < shape[0]; ++k) {
        float var_value = static_cast<float*>(var_res->get_data())[k];
        for (int i = 0; i < shape[1]; ++i) {
            float tmp = 0;
            for (int j = 0; j < shape[1]; ++j) {
                int eq = i == j;
                auto sigma = std::sqrt(var_value + eps);
                auto x_hat_i = norm_res_data[k * norm_res_strides[0] + i * norm_res_strides[1]];
                auto x_hat_j = norm_res_data[k * norm_res_strides[0] + j * norm_res_strides[1]];
                auto part1 = eq * shape[1] - 1 - x_hat_i * x_hat_j;
                auto part2 = shape[1] * sigma;
                auto g = part1 / part2;
                tmp += g * src_grad_data[k * src_grad_strides[0] + j * src_grad_strides[1]];
            }
            tgt_grad_data[k * tgt_grad_strides[0] + i * tgt_grad_strides[1]] = tmp;
        }
    }
}

void CPUOps::mulSV(Tensor* dst, Tensor* src, float value) {
    assert(dst->length() == src->length());
    auto length = dst->length();
    for (int i = 0; i < length; ++i) {
        static_cast<float*>(dst->get_data())[i] =
            static_cast<float*>(src->get_data())[i] * value;
    }
}

void* CPUOps::alloc(size_t size) {
    return malloc(size);
}

void CPUOps::memset(void* ptr, int value, size_t size) {
    ::memset(ptr, value, size);
}

void CPUOps::cp_device_to_device(void* dst, const void* src, size_t size) {
    ::memcpy(dst, src, size);
}

void CPUOps::free(void* ptr) {
    ::free(ptr);
}

void CPUOps::cp_to_device(Tensor* dst_tensor, char* src, size_t size) {
    assert(dst_tensor != nullptr);
    assert(src != nullptr);
    assert(size > 0);
    assert(dst_tensor->get_data() != nullptr);
    assert(dst_tensor->size() == size);
    memcpy(dst_tensor->get_data(), src, size);
}

void CPUOps::cp_from_device(char* dst, const Tensor* src_tensor, size_t size) {
    assert(dst != nullptr);
    assert(src_tensor != nullptr);
    assert(size > 0);
    assert(src_tensor->get_data() != nullptr);
    memcpy(dst, src_tensor->get_data(), size);
}