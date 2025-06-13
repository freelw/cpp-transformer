#ifndef GCC_CPU
#ifdef METAL_GPU

#include "metal_ops.h"
#include <string.h>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <vector>
#include <type_traits>

#define KERNEL_PATH "/backends/gpu/metal/kernel.metal"

MetalOps::MetalOps() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal device not found!" << std::endl;
        throw std::runtime_error("Metal device not found");
    }
    commandQueue = device->newCommandQueue();
    bufferArgs = device->newBuffer(128, MTL::ResourceStorageModeShared);
    load_kernel_metal();

    add_kernel(new MetalKops("tensor_add_2d", library));
}

MetalOps::~MetalOps() {
    for (auto& kernel : kernels) {
        delete kernel;
    }
    bufferArgs->release();
    commandQueue->release();
    device->release();
}

void MetalOps::add(
    Tensor* lhs, const Tensor* rhs, Tensor* res,
    Tensor* l_shape, Tensor* l_strides,
    Tensor* r_striedes, Tensor* res_striedes
) {

    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(l_shape != nullptr);
    assert(l_strides != nullptr);
    assert(r_striedes != nullptr);
    assert(res_striedes != nullptr);

    auto length = lhs->length();
}

void MetalOps::addEq(
    Tensor* lhs, const Tensor* rhs,
    Tensor* l_shape,
    Tensor* l_strides, Tensor* r_striedes
) {
    std::cerr << "Warning: 'addEq' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::expandAdd(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    std::cerr << "Warning: 'expandAdd' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::expandMul(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    std::cerr << "Warning: 'expandMul' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::at(Tensor* lhs, const Tensor* rhs, Tensor* res) {
    std::cerr << "Warning: 'at' operation is not implemented in MetalOps." << std::endl;
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
    std::cerr << "Warning: 'mul' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::sum(Tensor* lhs, Tensor* res, int dim) {
    std::cerr << "Warning: 'sum' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::relu(Tensor* lhs, Tensor* res) {
    std::cerr << "Warning: 'relu' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::reluPrime(Tensor* lhs, Tensor* res) {
    std::cerr << "Warning: 'reluPrime' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::crossEntropy(
    Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res
) {
    std::cerr << "Warning: 'crossEntropy' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::crossEntropyBackward(
    Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res
) {
    std::cerr << "Warning: 'crossEntropyBackward' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::calcAllGradNorm(const std::vector<Tensor*>& grads, Tensor* norm) {
    std::cerr << "Warning: 'calcAllGradNorm' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::clipGrad(Tensor* grad, const Tensor* norm, float grad_clip_val) {
    std::cerr << "Warning: 'clipGrad' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::adamStep(
    Tensor* w, Tensor* grad, Tensor* m, Tensor* v, int t, float lr, float beta1, float beta2, float epsilon
) {
    std::cerr << "Warning: 'adamStep' operation is not implemented in MetalOps." << std::endl;
}

void MetalOps::init_weight_gauss(Tensor* tensor, float mean, float sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<float> distribution_w(0.0, sigma);
    float* data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = distribution_w(generator_w) + mean;
    }
}

void MetalOps::init_weight_uniform(Tensor* tensor, float sigma) {
    assert(false);
}

void MetalOps::init_weight_for_dbg(Tensor* tensor, float scale) {
    assert(false);
}

void MetalOps::fill(Tensor* tensor, float value) {
    std::cerr << "Warning: 'fill' operation is not implemented in MetalOps." << std::endl;
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
    std::cerr << "Warning: 'div' operation is not implemented in MetalOps." << std::endl;
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

void* MetalOps::alloc(size_t size, void** ctx) {
    MTL::Buffer* buffer = device->newBuffer(size, MTL::ResourceStorageModeShared);
    *ctx = (void*)buffer;
    return buffer->contents();
}

void MetalOps::memset(void* ptr, int value, size_t size) {
    ::memset(ptr, value, size);
}

void MetalOps::free(void* ptr) {
    std::cerr << "Warning: Freeing memory in MetalOps is not implemented." << std::endl;
}

void MetalOps::cp_device_to_device(void* dst, const void* src, size_t size) {
    ::memcpy(dst, src, size);
}

void MetalOps::cp_to_device(Tensor* dst_tensor, char* src, size_t size) {
    assert(dst_tensor != nullptr);
    assert(src != nullptr);
    assert(size > 0);
    assert(dst_tensor->get_data() != nullptr);
    assert(dst_tensor->size() == size);
    memcpy(dst_tensor->get_data(), src, size);
}

void MetalOps::cp_from_device(char* dst, const Tensor* src_tensor, size_t size) {
    assert(dst != nullptr);
    assert(src_tensor != nullptr);
    assert(size > 0);
    assert(src_tensor->get_data() != nullptr);
    memcpy(dst, src_tensor->get_data(), size);
}

void MetalOps::load_kernel_metal() {
    char* s = getcwd(NULL, 0);
    std::string path = s;
    free(s);
    path += KERNEL_PATH;

    //load shader source code into shaderSource
    std::ifstream kernel_ifs(path, std::ios::binary);
    std::cout << "path: " << path << std::endl;
    shaderSource = std::string(std::istreambuf_iterator<char>(kernel_ifs), std::istreambuf_iterator<char>());
    std::cout << "shaderSource: " << shaderSource << std::endl;
    NS::Error* error = nullptr;
    library = device->newLibrary(NS::String::string(shaderSource.c_str(), NS::StringEncoding::UTF8StringEncoding), nullptr, &error);

    if (!library) {
        std::cerr << "Error compiling shader : " << error->localizedDescription()->utf8String() << std::endl;
        abort();
    }
}

void MetalOps::add_kernel(MetalKops* kernel) {
    assert(kernel != nullptr);
    kernels.push_back(kernel);
}

#endif // METAL_GPU
#endif // GCC_CPU