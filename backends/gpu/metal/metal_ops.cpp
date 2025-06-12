#ifndef GCC_CPU
#ifdef METAL_GPU

#include "metal_ops.h"

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <vector>
#include <type_traits>

MetalOps::MetalOps() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal device not found!" << std::endl;
        throw std::runtime_error("Metal device not found");
    }
    commandQueue = device->newCommandQueue();
}

MetalOps::~MetalOps() {
    commandQueue->release();
    device->release();
}

void MetalOps::add(
    Tensor* lhs, const Tensor* rhs, Tensor* res,
    Tensor* l_shape, Tensor* l_strides,
    Tensor* r_striedes, Tensor* res_striedes
) {
    assert(false);
}

void MetalOps::addEq(
    Tensor* lhs, const Tensor* rhs,
    Tensor* l_shape,
    Tensor* l_strides, Tensor* r_striedes
) {
    assert(false);
}

void MetalOps::expandAdd(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    assert(false);
}

void MetalOps::expandMul(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    assert(false);
}

void MetalOps::at(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    assert(false);
}

void MetalOps::embedding(Tensor* lhs, const Tensor* indices, const Tensor* res) {
    assert(false);
}

void MetalOps::embeddingBackward(Tensor* lhs, const Tensor* indices, Tensor* res) {
    assert(false);
}

void MetalOps::mul(
    Tensor* lhs, const Tensor* rhs, Tensor* res,
    Tensor* l_shape, Tensor* l_strides,
    Tensor* r_striedes, Tensor* res_striedes
) {
    assert(false);
}

void MetalOps::sum(Tensor* lhs, Tensor* res, int dim) {
    assert(false);
}

void MetalOps::relu(Tensor* lhs, Tensor* res) {
    assert(false);
}

void MetalOps::reluPrime(Tensor* lhs, Tensor* res) {
    assert(false);
}

void MetalOps::crossEntropy(
    Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res
) {
    assert(false);
}

void MetalOps::crossEntropyBackward(
    Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res
) {
    assert(false);
}

void MetalOps::calcAllGradNorm(const std::vector<Tensor*>& grads, Tensor* norm) {
    assert(false);
}

void MetalOps::clipGrad(Tensor* grad, const Tensor* norm, float grad_clip_val) {
    assert(false);
}

void MetalOps::adamStep(
    Tensor* w, Tensor* grad, Tensor* m, Tensor* v, int t, float lr, float beta1, float beta2, float epsilon
) {
    assert(false);
}

void MetalOps::init_weight_gauss(Tensor* tensor, float mean, float sigma) {
    assert(false);
}

void MetalOps::init_weight_uniform(Tensor* tensor, float sigma) {
    assert(false);
}

void MetalOps::init_weight_for_dbg(Tensor* tensor, float scale) {
    assert(false);
}

void MetalOps::fill(Tensor* tensor, float value) {
    assert(false);
}

void MetalOps::reshape_deep_cp(
    Tensor* dst_tensor, const Tensor* src_tensor,
    const Tensor* src_shape, const Tensor* src_strides
) {
    assert(false);
}

void MetalOps::repeat_interleave(Tensor* lhs, Tensor* res, int n) {
    assert(false);
}

void MetalOps::sequence_mask(Tensor* lhs, const Tensor* mask, Tensor* res, float value) {
    assert(false);
}

void MetalOps::softmax(Tensor* lhs, Tensor* res) {
    assert(false);
}

void MetalOps::softmax_bacward(Tensor* target_grad, const Tensor* softmax_res, Tensor* grad) {
    assert(false);
}

void MetalOps::div(Tensor* dst, Tensor* src, float value) {
    assert(false);
}

void MetalOps::build_dropout_mask(
    Tensor* mask, float p,
    Tensor* shape, Tensor* strides
) {
    assert(false);
}

void MetalOps::pos_encoding(Tensor* res) {
    assert(false);
}

void MetalOps::avg(Tensor* lhs, Tensor* res) {
    assert(false);
}

void MetalOps::var(Tensor* lhs, const Tensor* _avg, Tensor* res) {
    assert(false);
}

void MetalOps::norm(const Tensor* src, const Tensor* avg, const Tensor* var, Tensor* res) {
    assert(false);
}

void MetalOps::normBackward(
    const Tensor* src_grad, const Tensor* norm_res, const Tensor* var_res, Tensor* tgt_grad
) {
    assert(false);
}

void MetalOps::mulSV(Tensor* dst, Tensor* src, float value) {
    assert(false);
}

void* MetalOps::alloc(size_t size) {
    // Implement memory allocation using Metal
    assert(false);
    return nullptr;
}

void MetalOps::memset(void* ptr, int value, size_t size) {
    // Implement memory set using Metal
    assert(false);
}

void MetalOps::free(void* ptr) {
    // Implement memory deallocation using Metal
    assert(false);
}

void MetalOps::cp_device_to_device(void* dst, const void* src, size_t size) {
    // Implement device-to-device copy using Metal
    assert(false);
}

void MetalOps::cp_to_device(Tensor* dst_tensor, char* src, size_t size) {
    // Implement copy to device using Metal
    assert(false);
}

void MetalOps::cp_from_device(char* dst, const Tensor* src_tensor, size_t size) {
    // Implement copy from device using Metal
    assert(false);
}

void MetalOps::commit() {

}

void MetalOps::wait() {

}

#endif // METAL_GPU
#endif // GCC_CPU