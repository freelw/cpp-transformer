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
    bufferIntArgs = device->newBuffer(128, MTL::ResourceStorageModeShared);
    bufferFloatArgs = device->newBuffer(128, MTL::ResourceStorageModeShared);
    load_kernel_metal();

    addOps = new MetalKops("tensor_add_kernel", library);
    fillOps = new MetalKops("fill_float", library);
    atOps = new MetalKops("tensor_at_2d", library);
}

MetalOps::~MetalOps() {
    delete fillOps;
    delete addOps;
    bufferFloatArgs->release();
    bufferIntArgs->release();
    commandQueue->release();
    device->release();
}

int calc_offset(const Tensor* t) {
    char* base = reinterpret_cast<char*>(reinterpret_cast<MTL::Buffer*>(t->get_storage()->ctx)->contents());
    char* pos = reinterpret_cast<char*>(t->get_data());
    auto offset_res = pos - base;
    return offset_res;
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

    addOps->prepare(device, commandQueue);

    int* args = (int*)bufferIntArgs->contents();
    args[0] = lhs->get_dim();
    args[1] = lhs->length();
    auto offset_res = calc_offset(res);
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_l_shape = calc_offset(l_shape);
    auto offset_res_striedes = calc_offset(res_striedes);
    auto offset_l_strides = calc_offset(l_strides);
    auto offset_r_striedes = calc_offset(r_striedes);

    auto encoder = addOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_shape->get_storage()->ctx), offset_l_shape, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res_striedes->get_storage()->ctx), offset_res_striedes, 4);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_strides->get_storage()->ctx), offset_l_strides, 5);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(r_striedes->get_storage()->ctx), offset_r_striedes, 6);
    encoder->setBuffer(bufferIntArgs, 0, 7);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);

    addOps->run();
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

    this->memset((float*)res->get_data(), 0, res->size());

    atOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = M;
    args[1] = N;
    args[2] = P;
    args[3] = stride_M0;
    args[4] = stride_M1;
    args[5] = stride_N0;
    args[6] = stride_N1;
    args[7] = stride_P0;
    args[8] = stride_P1;
    auto offset_res = calc_offset(res);
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);

    auto encoder = atOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, 0, 3);

    MTL::Size gridDim = MTL::Size((P + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);

    encoder->dispatchThreadgroups(gridDim, blockDim);
    atOps->run();
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
    assert(tensor != nullptr);
    assert(tensor->get_data() != nullptr);
    assert(tensor->length() > 0);

    fillOps->prepare(device, commandQueue);
    int* argsInt = (int*)bufferIntArgs->contents();
    float* argsFloat = (float*)bufferFloatArgs->contents();
    auto length = tensor->length();
    argsInt[0] = length;
    argsFloat[0] = value;
    auto offset_tensor = calc_offset(tensor);
    auto encoder = fillOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(tensor->get_storage()->ctx), offset_tensor, 0);
    encoder->setBuffer(bufferIntArgs, 0, 1);
    encoder->setBuffer(bufferFloatArgs, 0, 2);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    fillOps->run();
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
    // std::cout << "shaderSource: " << shaderSource << std::endl;
    NS::Error* error = nullptr;
    library = device->newLibrary(NS::String::string(shaderSource.c_str(), NS::StringEncoding::UTF8StringEncoding), nullptr, &error);

    if (!library) {
        std::cerr << "Error compiling shader : " << error->localizedDescription()->utf8String() << std::endl;
        abort();
    }
}

#endif // METAL_GPU
#endif // GCC_CPU