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
    addEqOps = new MetalKops("tensor_add_eq_kernel", library);
    expandAddOps = new MetalKops("expand_add", library);
    mulOps = new MetalKops("tensor_mul_kernel", library);
    sumOps = new MetalKops("tensor_sum_2d_dim0_v1", library);
    crossEntropyOps = new MetalKops("cross_entropy", library);
    crossEntropyBackwardOps = new MetalKops("cross_entropy_backward", library);
    reluOps = new MetalKops("tensor_relu", library);
    divOps = new MetalKops("tensor_div_scalar", library);
    expandMulOps = new MetalKops("expand_mul", library);
    reluPrimeOps = new MetalKops("tensor_relu_prime", library);
    calcAllGradNormOps = new MetalKops("tensor_l2_norm", library);
    clipGradOps = new MetalKops("tensor_clip", library);
    adamStepOps = new MetalKops("tensor_adam_step", library);
    reshapeDeepCpOps = new MetalKops("reshape_deep_cp_float_kernel", library);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = std::mt19937(seed);
    dis = std::uniform_real_distribution<>(0, 1);
}

MetalOps::~MetalOps() {
    delete reshapeDeepCpOps;
    delete adamStepOps;
    delete clipGradOps;
    delete calcAllGradNormOps;
    delete reluPrimeOps;
    delete expandMulOps;
    delete divOps;
    delete reluOps;
    delete crossEntropyBackwardOps;
    delete crossEntropyOps;
    delete sumOps;
    delete mulOps;
    delete expandAddOps;
    delete addEqOps;
    delete atOps;
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
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    auto lshape = lhs->get_shape();
    auto rshape = rhs->get_shape();
    assert(lshape == rshape);
    int dim = lhs->get_dim();
    auto length = lhs->length();

    addEqOps->prepare(device, commandQueue);

    int* args = (int*)bufferIntArgs->contents();
    args[0] = dim;
    args[1] = length;
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_l_shape = calc_offset(l_shape);
    auto offset_l_strides = calc_offset(l_strides);
    auto offset_r_striedes = calc_offset(r_striedes);

    auto encoder = addEqOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_shape->get_storage()->ctx), offset_l_shape, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_strides->get_storage()->ctx), offset_l_strides, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(r_striedes->get_storage()->ctx), offset_r_striedes, 4);
    encoder->setBuffer(bufferIntArgs, 0, 5);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    addEqOps->run();
}

void MetalOps::expandAdd(Tensor* lhs, const Tensor* rhs, Tensor* res) {
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

    expandAddOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = shape[0];
    args[1] = shape[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res_strides[0];
    args[5] = res_strides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_res = calc_offset(res);
    auto encoder = expandAddOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, 0, 3);

    MTL::Size gridDim = MTL::Size(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    expandAddOps->run();
}

void MetalOps::expandMul(Tensor* lhs, const Tensor* rhs, Tensor* res) {
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

    expandMulOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = shape[0];
    args[1] = shape[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res_strides[0];
    args[5] = res_strides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_res = calc_offset(res);
    auto encoder = expandMulOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, 0, 3);

    MTL::Size gridDim = MTL::Size(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    expandMulOps->run();
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
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(l_shape != nullptr);
    assert(l_strides != nullptr);
    assert(r_striedes != nullptr);
    assert(res_striedes != nullptr);

    auto length = lhs->length();

    int* args = (int*)bufferIntArgs->contents();
    args[0] = lhs->get_dim();
    args[1] = length;
    auto offset_res = calc_offset(res);
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_l_shape = calc_offset(l_shape);
    auto offset_res_striedes = calc_offset(res_striedes);
    auto offset_l_strides = calc_offset(l_strides);
    auto offset_r_striedes = calc_offset(r_striedes);
    mulOps->prepare(device, commandQueue);
    auto encoder = mulOps->getEncoder();
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
    mulOps->run();
}

void MetalOps::sum(Tensor* lhs, Tensor* res, int dim) {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(dim >= 0 && dim < lhs->get_dim());

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(dim == 0);
    auto lstrides = lhs->get_strides();
    assert(lhs->get_dim() == 2);
    assert(res->get_dim() == 1);

    sumOps->prepare(device, commandQueue);

    int* args = (int*)bufferIntArgs->contents();
    args[0] = shape[0];
    args[1] = shape[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res->get_strides()[0];

    auto offset_src = calc_offset(lhs);
    auto offset_res = calc_offset(res);

    auto encoder = sumOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_src, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, 0, 2);

    size_t sharedMemorySize = TILE_WIDTH * sizeof(float); // Calculate shared memory size
    encoder->setThreadgroupMemoryLength(sharedMemorySize, 0); // Set shared memory size

    MTL::Size gridDim = MTL::Size(
        shape[1],
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(1, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    sumOps->run();
}

void MetalOps::relu(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    reluOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = length;

    auto offset_lhs = calc_offset(lhs);
    auto offset_res = calc_offset(res);
    auto encoder = reluOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, 0, 2);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    reluOps->run();
}

void MetalOps::reluPrime(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    reluPrimeOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = length;
    auto offset_lhs = calc_offset(lhs);
    auto offset_res = calc_offset(res);
    auto encoder = reluPrimeOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, 0, 2);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    reluPrimeOps->run();
}

void MetalOps::crossEntropy(
    Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res
) {
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

    this->memset((float*)res->get_data(), 0, res->size());
    this->memset((float*)maxs->get_data(), 0, maxs->size());
    this->memset((float*)sums->get_data(), 0, sums->size());

    crossEntropyOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = lhs->get_shape()[0];
    args[1] = lhs->get_shape()[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_labels = calc_offset(labels);
    auto offset_maxs = calc_offset(maxs);
    auto offset_sums = calc_offset(sums);
    auto offset_res = calc_offset(res);
    auto encoder = crossEntropyOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(labels->get_storage()->ctx), offset_labels, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(maxs->get_storage()->ctx), offset_maxs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(sums->get_storage()->ctx), offset_sums, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 4);
    encoder->setBuffer(bufferIntArgs, 0, 5);
    MTL::Size gridDim = MTL::Size((lhs->get_shape()[0] + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    crossEntropyOps->run();
}

void MetalOps::crossEntropyBackward(
    Tensor* lhs, const Tensor* labels, Tensor* maxs, Tensor* sums, Tensor* res
) {
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

    crossEntropyBackwardOps->prepare(device, commandQueue);
    int* args = (int*)bufferIntArgs->contents();
    args[0] = batch_size;
    args[1] = size;
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res_strides[0];
    args[5] = res_strides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_labels = calc_offset(labels);
    auto offset_maxs = calc_offset(maxs);
    auto offset_sums = calc_offset(sums);
    auto offset_res = calc_offset(res);
    auto encoder = crossEntropyBackwardOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(labels->get_storage()->ctx), offset_labels, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(maxs->get_storage()->ctx), offset_maxs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(sums->get_storage()->ctx), offset_sums, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 4);
    encoder->setBuffer(bufferIntArgs, 0, 5);

    MTL::Size gridDim = MTL::Size((batch_size + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    crossEntropyBackwardOps->run();
}

void MetalOps::calcAllGradNorm(const std::vector<Tensor*>& grads, Tensor* norm) {
    assert(norm != nullptr);
    assert(norm->length() == 1);
    this->memset((float*)norm->get_data(), 0, norm->size());
    for (auto& grad : grads) {
        assert(grad != nullptr);
        auto length = grad->length();

        calcAllGradNormOps->prepare(device, commandQueue);
        int* args = (int*)bufferIntArgs->contents();
        args[0] = length;
        auto offset_grad = calc_offset(grad);
        auto offset_norm = calc_offset(norm);

        auto encoder = calcAllGradNormOps->getEncoder();
        assert(encoder != nullptr);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 0);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(norm->get_storage()->ctx), offset_norm, 1);
        encoder->setBuffer(bufferIntArgs, 0, 2);
        size_t sharedMemorySize = TILE_WIDTH * sizeof(float); // Calculate shared memory size
        encoder->setThreadgroupMemoryLength(sharedMemorySize, 0); // Set shared memory size
        MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
        MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
        encoder->dispatchThreadgroups(gridDim, blockDim);
        calcAllGradNormOps->run();
    }
}

void MetalOps::clipGrad(Tensor* grad, const Tensor* norm, float grad_clip_val) {
    assert(grad != nullptr);
    assert(norm != nullptr);

    assert(norm->get_shape().size() == 1);

    auto length = grad->length();
    auto norm_length = norm->length();
    assert(norm_length == 1);

    clipGradOps->prepare(device, commandQueue);
    int* intArgs = (int*)bufferIntArgs->contents();
    float* floatArgs = (float*)bufferFloatArgs->contents();
    intArgs[0] = length;
    floatArgs[0] = grad_clip_val;
    auto offset_grad = calc_offset(grad);
    auto offset_norm = calc_offset(norm);
    auto encoder = clipGradOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(norm->get_storage()->ctx), offset_norm, 1);
    encoder->setBuffer(bufferIntArgs, 0, 2);
    encoder->setBuffer(bufferFloatArgs, 0, 3);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    clipGradOps->run();
}

void MetalOps::adamStep(
    Tensor* w, Tensor* grad, Tensor* m, Tensor* v, int t, float lr, float beta1, float beta2, float epsilon
) {
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

    adamStepOps->prepare(device, commandQueue);
    int* intArgs = (int*)bufferIntArgs->contents();
    float* floatArgs = (float*)bufferFloatArgs->contents();
    intArgs[0] = length;
    intArgs[1] = t;
    floatArgs[0] = beta1;
    floatArgs[1] = beta2;
    floatArgs[2] = lr;
    floatArgs[3] = epsilon;
    auto offset_w = calc_offset(w);
    auto offset_grad = calc_offset(grad);
    auto offset_m = calc_offset(m);
    auto offset_v = calc_offset(v);

    auto encoder = adamStepOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(w->get_storage()->ctx), offset_w, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(m->get_storage()->ctx), offset_m, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(v->get_storage()->ctx), offset_v, 3);
    encoder->setBuffer(bufferIntArgs, 0, 4);
    encoder->setBuffer(bufferFloatArgs, 0, 5);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    adamStepOps->run();
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
    auto size = tensor->size();
    void* _data = ::malloc(size);

    if (tensor->get_dtype() == FLOAT32) {
        float* data = static_cast<float*>(_data);
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = static_cast<float>(i) * 1e-5 * scale;
        }
    }
    else if (tensor->get_dtype() == INT32) {
        int32_t* data = static_cast<int32_t*>(_data);
        for (int i = 0; i < tensor->length(); ++i) {
            data[i] = i % 10;
        }
    }
    else {
        assert(false);
    }
    this->cp_to_device(tensor, (char*)_data, size);
    ::free(_data);
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
    }
    else if (dtype == FLOAT32) {
        reshapeDeepCpOps->prepare(device, commandQueue);
        int* args = (int*)bufferIntArgs->contents();
        args[0] = dim;
        args[1] = length;
        auto offset_dst = calc_offset(dst_tensor);
        auto offset_src = calc_offset(src_tensor);
        auto offset_src_shape = calc_offset(src_shape);
        auto offset_src_strides = calc_offset(src_strides);
        auto encoder = reshapeDeepCpOps->getEncoder();
        assert(encoder != nullptr);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(dst_tensor->get_storage()->ctx), offset_dst, 0);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src_tensor->get_storage()->ctx), offset_src, 1);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src_shape->get_storage()->ctx), offset_src_shape, 2);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src_strides->get_storage()->ctx), offset_src_strides, 3);
        encoder->setBuffer(bufferIntArgs, 0, 4);
        MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
        MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
        encoder->dispatchThreadgroups(gridDim, blockDim);
        reshapeDeepCpOps->run();
    }
    else {
        assert(false);
    }
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
    assert(dst->length() == src->length());
    assert(dst->get_shape() == src->get_shape());
    assert(dst->get_strides() == src->get_strides());
    auto length = dst->length();

    divOps->prepare(device, commandQueue);
    int* intArgs = (int*)bufferIntArgs->contents();
    float* floatArgs = (float*)bufferFloatArgs->contents();
    intArgs[0] = length;
    floatArgs[0] = value;
    auto offset_dst = calc_offset(dst);
    auto offset_src = calc_offset(src);
    auto encoder = divOps->getEncoder();
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(dst->get_storage()->ctx), offset_dst, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src->get_storage()->ctx), offset_src, 1);
    encoder->setBuffer(bufferIntArgs, 0, 2);
    encoder->setBuffer(bufferFloatArgs, 0, 3);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    divOps->run();
}

void MetalOps::build_dropout_mask(
    Tensor* mask, float p,
    Tensor* shape, Tensor* strides
) {
    // todo : use gpu
    assert(mask != nullptr);
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
    // we do not have to free the buffer here, as it is managed by MetalOps
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
    ::memcpy(dst_tensor->get_data(), src, size);
}

void MetalOps::cp_from_device(char* dst, const Tensor* src_tensor, size_t size) {
    assert(dst != nullptr);
    assert(src_tensor != nullptr);
    assert(size > 0);
    assert(src_tensor->get_data() != nullptr);
    ::memcpy(dst, src_tensor->get_data(), size);
}

void MetalOps::load_kernel_metal() {
    char* s = getcwd(NULL, 0);
    std::string path = s;
    free(s);
    path += KERNEL_PATH;

    //load shader source code into shaderSource
    std::ifstream kernel_ifs(path, std::ios::binary);
    // std::cout << "path: " << path << std::endl;
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