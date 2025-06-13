#ifndef METAL_OPS_H
#define METAL_OPS_H

#include "backends/backend_ops.h"
#include "kernel_ops.h"

#ifndef GCC_CPU
#ifdef METAL_GPU

#define TILE_WIDTH 32

namespace MTL {
    class Device;
    class CommandQueue;
    class ComputePipelineState;
    class CommandBuffer;
    class ComputeCommandEncoder;
    class Buffer;
    class Library;
}

namespace NS {
    class Error;
}

class MetalOps : public BackendOps {
public:
    MetalOps();
    ~MetalOps();
    void add(
        Tensor* lhs, const Tensor* rhs, Tensor* res,
        Tensor* l_shape, Tensor* l_strides,
        Tensor* r_striedes, Tensor* res_striedes
    ) override;
    void addEq(
        Tensor* lhs, const Tensor* rhs,
        Tensor* l_shape,
        Tensor* l_strides, Tensor* r_striedes
    ) override;
    void expandAdd(Tensor* lhs, const Tensor* rhs, Tensor* res) override;
    void expandMul(Tensor* lhs, const Tensor* rhs, Tensor* res) override;
    void at(Tensor* lhs, const Tensor* rhs, Tensor* res) override;
    void embedding(Tensor* lhs, const Tensor* indices, const Tensor* res) override;
    void embeddingBackward(Tensor* lhs, const Tensor* indices, Tensor* res) override;
    void mul(
        Tensor* lhs, const Tensor* rhs, Tensor* res,
        Tensor* l_shape, Tensor* l_strides,
        Tensor* r_striedes, Tensor* res_striedes
    ) override;
    void sum(Tensor* lhs, Tensor* res, int dim) override;
    void relu(Tensor* lhs, Tensor* res) override;
    void reluPrime(Tensor* lhs, Tensor* res) override;
    void crossEntropy(Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res) override;
    void crossEntropyBackward(Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res) override;
    void calcAllGradNorm(const std::vector<Tensor*>& grads, Tensor* norm) override;
    void clipGrad(Tensor* grad, const Tensor* norm, float grad_clip_val) override;
    void adamStep(Tensor* w, Tensor* grad, Tensor* m, Tensor* v, int t, float lr, float beta1, float beta2, float epsilon) override;
    void init_weight_gauss(Tensor* tensor, float mean, float sigma) override;
    void init_weight_uniform(Tensor* tensor, float sigma) override;
    void init_weight_for_dbg(Tensor* tensor, float scale) override;
    void fill(Tensor* tensor, float value) override;
    void reshape_deep_cp(
        Tensor* dst_tensor, const Tensor* src_tensor,
        const Tensor* src_shape, const Tensor* src_strides
    ) override;
    void repeat_interleave(Tensor* lhs, Tensor* res, int n) override;
    void sequence_mask(Tensor* lhs, const Tensor* mask, Tensor* res, float value) override;
    void softmax(Tensor* lhs, Tensor* res) override;
    void softmax_bacward(Tensor* target_grad, const Tensor* softmax_res, Tensor* grad) override;
    void div(Tensor* dst, Tensor* src, float value) override;
    void build_dropout_mask(
        Tensor* mask, float p,
        Tensor* shape, Tensor* strides
    ) override;
    void pos_encoding(Tensor* res) override;
    void avg(Tensor* lhs, Tensor* res) override;
    void var(Tensor* lhs, const Tensor* _avg, Tensor* res) override;
    void norm(const Tensor* src, const Tensor* avg, const Tensor* var, Tensor* res) override;
    void normBackward(
        const Tensor* src_grad, const Tensor* norm_res, const Tensor* var_res, Tensor* tgt_grad
    ) override;
    void mulSV(Tensor* dst, Tensor* src, float value) override;

    // Memory management
    void* alloc(size_t size, void** ctx = nullptr) override;
    void memset(void* ptr, int value, size_t size) override;
    void free(void* ptr) override;
    void cp_device_to_device(void* dst, const void* src, size_t size) override;
    void cp_to_device(Tensor* dst_tensor, char* src, size_t size) override;
    void cp_from_device(char* dst, const Tensor* src_tensor, size_t size) override;
private:
    void load_kernel_metal();
private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;

    MTL::Buffer* bufferIntArgs;
    MTL::Buffer* bufferFloatArgs;
    MTL::Library* library;
    std::string shaderSource;
    MetalKops* addOps;
    MetalKops* fillOps;
    MetalKops* atOps;
};

#endif // METAL_GPU
#endif // GCC_CPU

#endif