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
#define TOTAL_INT_ARGS 4096
#define TOTAL_FLOAT_ARGS 2048

MetalOps::MetalOps() : commandBuffer(nullptr), cur_int_args(0), cur_float_args(0) {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal device not found!" << std::endl;
        throw std::runtime_error("Metal device not found");
    }
    commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        std::cerr << "Failed to create command queue!" << std::endl;
        throw std::runtime_error("Failed to create command queue");
    }

    bufferIntArgs = device->newBuffer(TOTAL_INT_ARGS * sizeof(int), MTL::ResourceStorageModeShared);
    bufferFloatArgs = device->newBuffer(TOTAL_FLOAT_ARGS * sizeof(float), MTL::ResourceStorageModeShared);
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
    repeatInterleaveOps = new MetalKops("repeat_interleave_int32_kernel", library);
    sequenceMaskOps = new MetalKops("sequence_mask_kernel", library);
    softmaxOps = new MetalKops("softmax_kernel", library);
    softmaxBackwardOps = new MetalKops("softmax_backward_kernel", library);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = std::mt19937(seed);
    dis = std::uniform_real_distribution<>(0, 1);
}

MetalOps::~MetalOps() {
    delete softmaxBackwardOps;
    delete softmaxOps;
    delete sequenceMaskOps;
    delete repeatInterleaveOps;
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

void MetalOps::prepare() {
    commandBuffer = commandQueue->commandBuffer();
    cur_int_args = 0;
    cur_float_args = 0;
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
    auto encoder = addOps->prepare(device, commandQueue, commandBuffer);

    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(2);
    args[0] = lhs->get_dim();
    args[1] = lhs->length();

    auto offset_res = calc_offset(res);
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_l_shape = calc_offset(l_shape);
    auto offset_res_striedes = calc_offset(res_striedes);
    auto offset_l_strides = calc_offset(l_strides);
    auto offset_r_striedes = calc_offset(r_striedes);

    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_shape->get_storage()->ctx), offset_l_shape, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res_striedes->get_storage()->ctx), offset_res_striedes, 4);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_strides->get_storage()->ctx), offset_l_strides, 5);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(r_striedes->get_storage()->ctx), offset_r_striedes, 6);
    encoder->setBuffer(bufferIntArgs, offset_args, 7);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = addEqOps->prepare(device, commandQueue, commandBuffer);

    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(2);
    args[0] = dim;
    args[1] = length;
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_l_shape = calc_offset(l_shape);
    auto offset_l_strides = calc_offset(l_strides);
    auto offset_r_striedes = calc_offset(r_striedes);

    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_shape->get_storage()->ctx), offset_l_shape, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_strides->get_storage()->ctx), offset_l_strides, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(r_striedes->get_storage()->ctx), offset_r_striedes, 4);
    encoder->setBuffer(bufferIntArgs, offset_args, 5);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = expandAddOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(6);
    args[0] = shape[0];
    args[1] = shape[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res_strides[0];
    args[5] = res_strides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, offset_args, 3);

    MTL::Size gridDim = MTL::Size(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = expandMulOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(6);
    args[0] = shape[0];
    args[1] = shape[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res_strides[0];
    args[5] = res_strides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, offset_args, 3);

    MTL::Size gridDim = MTL::Size(
        (shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = atOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(9);
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

    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, offset_args, 3);

    MTL::Size gridDim = MTL::Size((P + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);

    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(2);
    args[0] = lhs->get_dim();
    args[1] = length;
    auto offset_res = calc_offset(res);
    auto offset_lhs = calc_offset(lhs);
    auto offset_rhs = calc_offset(rhs);
    auto offset_l_shape = calc_offset(l_shape);
    auto offset_res_striedes = calc_offset(res_striedes);
    auto offset_l_strides = calc_offset(l_strides);
    auto offset_r_striedes = calc_offset(r_striedes);
    auto encoder = mulOps->prepare(device, commandQueue, commandBuffer);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(rhs->get_storage()->ctx), offset_rhs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_shape->get_storage()->ctx), offset_l_shape, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res_striedes->get_storage()->ctx), offset_res_striedes, 4);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(l_strides->get_storage()->ctx), offset_l_strides, 5);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(r_striedes->get_storage()->ctx), offset_r_striedes, 6);
    encoder->setBuffer(bufferIntArgs, offset_args, 7);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = sumOps->prepare(device, commandQueue, commandBuffer);

    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(5);
    args[0] = shape[0];
    args[1] = shape[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];
    args[4] = res->get_strides()[0];

    auto offset_src = calc_offset(lhs);
    auto offset_res = calc_offset(res);

    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_src, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, offset_args, 2);

    size_t sharedMemorySize = TILE_WIDTH * sizeof(float); // Calculate shared memory size
    encoder->setThreadgroupMemoryLength(sharedMemorySize, 0); // Set shared memory size

    MTL::Size gridDim = MTL::Size(
        shape[1],
        (shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(1, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::relu(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    auto encoder = reluOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(1);
    args[0] = length;

    auto offset_lhs = calc_offset(lhs);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, offset_args, 2);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::reluPrime(Tensor* lhs, Tensor* res) {
    assert(lhs != nullptr);
    assert(res != nullptr);

    int length = lhs->length();

    auto shape = lhs->get_shape();
    auto res_shape = res->get_shape();
    assert(shape == res_shape);

    auto encoder = reluPrimeOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(1);
    args[0] = length;
    auto offset_lhs = calc_offset(lhs);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, offset_args, 2);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = crossEntropyOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(4);
    args[0] = lhs->get_shape()[0];
    args[1] = lhs->get_shape()[1];
    args[2] = lstrides[0];
    args[3] = lstrides[1];

    auto offset_lhs = calc_offset(lhs);
    auto offset_labels = calc_offset(labels);
    auto offset_maxs = calc_offset(maxs);
    auto offset_sums = calc_offset(sums);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(labels->get_storage()->ctx), offset_labels, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(maxs->get_storage()->ctx), offset_maxs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(sums->get_storage()->ctx), offset_sums, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 4);
    encoder->setBuffer(bufferIntArgs, offset_args, 5);
    MTL::Size gridDim = MTL::Size((lhs->get_shape()[0] + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = crossEntropyBackwardOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(6);
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
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(labels->get_storage()->ctx), offset_labels, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(maxs->get_storage()->ctx), offset_maxs, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(sums->get_storage()->ctx), offset_sums, 3);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 4);
    encoder->setBuffer(bufferIntArgs, offset_args, 5);

    MTL::Size gridDim = MTL::Size((batch_size + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::calcAllGradNorm(const std::vector<Tensor*>& grads, Tensor* norm) {
    assert(norm != nullptr);
    assert(norm->length() == 1);
    this->memset((float*)norm->get_data(), 0, norm->size());
    for (auto& grad : grads) {
        assert(grad != nullptr);
        auto length = grad->length();

        auto encoder = calcAllGradNormOps->prepare(device, commandQueue, commandBuffer);
        auto offset_args = cur_int_args * sizeof(int);
        int* args = get_cur_int_args_buffer(1);
        args[0] = length;
        auto offset_grad = calc_offset(grad);
        auto offset_norm = calc_offset(norm);

        assert(encoder != nullptr);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 0);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(norm->get_storage()->ctx), offset_norm, 1);
        encoder->setBuffer(bufferIntArgs, offset_args, 2);
        size_t sharedMemorySize = TILE_WIDTH * sizeof(float); // Calculate shared memory size
        encoder->setThreadgroupMemoryLength(sharedMemorySize, 0); // Set shared memory size
        MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
        MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
        encoder->dispatchThreadgroups(gridDim, blockDim);
        encoder->endEncoding();
        encoder->release();
    }
}

void MetalOps::clipGrad(Tensor* grad, const Tensor* norm, float grad_clip_val) {
    assert(grad != nullptr);
    assert(norm != nullptr);

    assert(norm->get_shape().size() == 1);

    auto length = grad->length();
    auto norm_length = norm->length();
    assert(norm_length == 1);

    auto encoder = clipGradOps->prepare(device, commandQueue, commandBuffer);
    auto offset_IntArgs = cur_int_args * sizeof(int);
    int* intArgs = get_cur_int_args_buffer(1);
    auto offset_FloatArgs = cur_float_args * sizeof(float);
    float* floatArgs = get_cur_float_args_buffer(1);
    intArgs[0] = length;
    floatArgs[0] = grad_clip_val;
    auto offset_grad = calc_offset(grad);
    auto offset_norm = calc_offset(norm);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(norm->get_storage()->ctx), offset_norm, 1);
    encoder->setBuffer(bufferIntArgs, offset_IntArgs, 2);
    encoder->setBuffer(bufferFloatArgs, offset_FloatArgs, 3);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    auto encoder = adamStepOps->prepare(device, commandQueue, commandBuffer);
    auto offset_IntArgs = cur_int_args * sizeof(int);
    int* intArgs = get_cur_int_args_buffer(2);
    auto offset_FloatArgs = cur_float_args * sizeof(float);
    float* floatArgs = get_cur_float_args_buffer(4);
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

    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(w->get_storage()->ctx), offset_w, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(m->get_storage()->ctx), offset_m, 2);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(v->get_storage()->ctx), offset_v, 3);
    encoder->setBuffer(bufferIntArgs, offset_IntArgs, 4);
    encoder->setBuffer(bufferFloatArgs, offset_FloatArgs, 5);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

    assert(tensor->get_dtype() == FLOAT32);
    float* data = static_cast<float*>(tensor->get_data());
    for (int i = 0; i < tensor->length(); ++i) {
        data[i] = value;
    }

    // fillOps->prepare(device, commandQueue, commandBuffer, encoder);
    // int* argsInt = (int*)bufferIntArgs->contents();
    // float* argsFloat = (float*)bufferFloatArgs->contents();
    // auto length = tensor->length();
    // argsInt[0] = length;
    // argsFloat[0] = value;
    // auto offset_tensor = calc_offset(tensor);

    // assert(encoder != nullptr);
    // encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(tensor->get_storage()->ctx), offset_tensor, 0);
    // encoder->setBuffer(bufferIntArgs, 0, 1);
    // encoder->setBuffer(bufferFloatArgs, 0, 2);
    // MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    // MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    // encoder->dispatchThreadgroups(gridDim, blockDim);
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
        auto encoder = reshapeDeepCpOps->prepare(device, commandQueue, commandBuffer);
        auto offset_args = cur_int_args * sizeof(int);
        int* args = get_cur_int_args_buffer(2);
        args[0] = dim;
        args[1] = length;
        auto offset_dst = calc_offset(dst_tensor);
        auto offset_src = calc_offset(src_tensor);
        auto offset_src_shape = calc_offset(src_shape);
        auto offset_src_strides = calc_offset(src_strides);
        assert(encoder != nullptr);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(dst_tensor->get_storage()->ctx), offset_dst, 0);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src_tensor->get_storage()->ctx), offset_src, 1);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src_shape->get_storage()->ctx), offset_src_shape, 2);
        encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src_strides->get_storage()->ctx), offset_src_strides, 3);
        encoder->setBuffer(bufferIntArgs, offset_args, 4);
        MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
        MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
        encoder->dispatchThreadgroups(gridDim, blockDim);
        encoder->endEncoding();
        encoder->release();
    }
    else {
        assert(false);
    }
}

void MetalOps::repeat_interleave(Tensor* lhs, Tensor* res, int n) {
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
    }
    else {
        width = lshape[dim - 1];
    }
    auto l_length = lhs->length();
    auto r_length = res->length();
    assert(l_length * n == r_length);
    assert(l_length % width == 0);

    auto encoder = repeatInterleaveOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(4);
    args[0] = width;
    args[1] = l_length;
    args[2] = r_length;
    args[3] = n;
    auto offset_lhs = calc_offset(lhs);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, offset_args, 2);
    MTL::Size gridDim = MTL::Size((r_length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::sequence_mask(Tensor* lhs, const Tensor* mask, Tensor* res, float value) {
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

    auto encoder = sequenceMaskOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* intArgs = get_cur_int_args_buffer(7);
    auto offset_floatArgs = cur_float_args * sizeof(float);
    float* floatArgs = get_cur_float_args_buffer(1);
    intArgs[0] = lshape[0]; // batch size
    intArgs[1] = lshape[1]; // sequence length
    intArgs[2] = lstrides[0]; // mask length
    intArgs[3] = lstrides[1]; // lhs stride
    intArgs[4] = mstrides[0]; // mask stride
    intArgs[5] = rstrides[0]; // res stride
    intArgs[6] = rstrides[1]; // res stride
    floatArgs[0] = value; // fill value
    auto offset_lhs = calc_offset(lhs);
    auto offset_mask = calc_offset(mask);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(mask->get_storage()->ctx), offset_mask, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 2);
    encoder->setBuffer(bufferIntArgs, offset_args, 3);
    encoder->setBuffer(bufferFloatArgs, offset_floatArgs, 4);
    MTL::Size gridDim = MTL::Size(
        (lshape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (lshape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::softmax(Tensor* lhs, Tensor* res) {
    auto l_shape = lhs->get_shape();
    auto r_shape = res->get_shape();
    assert(l_shape == r_shape);
    assert(lhs->get_dtype() == FLOAT32);
    assert(res->get_dtype() == FLOAT32);
    assert(lhs->get_dim() == 3);
    assert(res->get_dim() == 3);
    auto lstrides = lhs->get_strides();
    auto rstrides = res->get_strides();

    auto encoder = softmaxOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(9);
    args[0] = l_shape[0]; // batch size
    args[1] = l_shape[1]; // sequence length
    args[2] = l_shape[2]; // feature size
    args[3] = lstrides[0]; // lhs stride
    args[4] = lstrides[1]; // lhs stride
    args[5] = lstrides[2]; // lhs stride
    args[6] = rstrides[0]; // res stride
    args[7] = rstrides[1]; // res stride
    args[8] = rstrides[2]; // res stride
    auto offset_lhs = calc_offset(lhs);
    auto offset_res = calc_offset(res);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(lhs->get_storage()->ctx), offset_lhs, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(res->get_storage()->ctx), offset_res, 1);
    encoder->setBuffer(bufferIntArgs, offset_args, 2);
    MTL::Size gridDim = MTL::Size(
        (l_shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (l_shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::softmax_bacward(Tensor* target_grad, const Tensor* softmax_res, Tensor* grad) {
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

    auto encoder = softmaxBackwardOps->prepare(device, commandQueue, commandBuffer);
    auto offset_args = cur_int_args * sizeof(int);
    int* args = get_cur_int_args_buffer(12);
    args[0] = t_shape[0]; // batch size
    args[1] = t_shape[1]; // sequence length
    args[2] = t_shape[2]; // feature size
    args[3] = t_strides[0]; // target_grad stride
    args[4] = t_strides[1]; // target_grad stride
    args[5] = t_strides[2]; // target_grad stride
    args[6] = s_strides[0]; // softmax_res stride
    args[7] = s_strides[1]; // softmax_res stride
    args[8] = s_strides[2]; // softmax_res stride
    args[9] = g_strides[0]; // grad stride
    args[10] = g_strides[1]; // grad stride
    args[11] = g_strides[2]; // grad stride
    auto offset_target_grad = calc_offset(target_grad);
    auto offset_softmax_res = calc_offset(softmax_res);
    auto offset_grad = calc_offset(grad);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(target_grad->get_storage()->ctx), offset_target_grad, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(softmax_res->get_storage()->ctx), offset_softmax_res, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(grad->get_storage()->ctx), offset_grad, 2);
    encoder->setBuffer(bufferIntArgs, offset_args, 3);
    MTL::Size gridDim = MTL::Size(
        (t_shape[1] + TILE_WIDTH - 1) / TILE_WIDTH,
        (t_shape[0] + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, TILE_WIDTH, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
}

void MetalOps::div(Tensor* dst, Tensor* src, Tensor* value) {
    assert(dst->length() == src->length());
    assert(dst->get_shape() == src->get_shape());
    assert(dst->get_strides() == src->get_strides());
    assert(value->length() == 1);
    auto length = dst->length();
    auto encoder = divOps->prepare(device, commandQueue, commandBuffer);
    auto offset_IntArgs = cur_int_args * sizeof(int);
    int* intArgs = get_cur_int_args_buffer(1);
    auto offset_FloatArgs = cur_float_args * sizeof(float);
    intArgs[0] = length;

    auto offset_dst = calc_offset(dst);
    auto offset_src = calc_offset(src);
    auto offset_value = calc_offset(value);
    assert(encoder != nullptr);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(dst->get_storage()->ctx), offset_dst, 0);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(src->get_storage()->ctx), offset_src, 1);
    encoder->setBuffer(reinterpret_cast<MTL::Buffer*>(value->get_storage()->ctx), offset_value, 2);
    encoder->setBuffer(bufferIntArgs, offset_IntArgs, 3);
    MTL::Size gridDim = MTL::Size((length + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);
    MTL::Size blockDim = MTL::Size(TILE_WIDTH, 1, 1);
    encoder->dispatchThreadgroups(gridDim, blockDim);
    encoder->endEncoding();
    encoder->release();
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

void MetalOps::commit() {
    commandBuffer->commit();
}

void MetalOps::wait() {
    commandBuffer->waitUntilCompleted();
    commandBuffer->release();
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

int* MetalOps::get_cur_int_args_buffer(int size) {
    if (cur_int_args + size >= TOTAL_INT_ARGS) {
        std::cerr << "cur_int_args + size >= TOTAL_INT_ARGS" << std::endl;
        abort();
    }
    int* ret = reinterpret_cast<int*>(bufferIntArgs->contents()) + cur_int_args;
    cur_int_args += size;
    return ret;
}

float* MetalOps::get_cur_float_args_buffer(int size) {
    if (cur_float_args + size >= TOTAL_FLOAT_ARGS) {
        std::cerr << "cur_float_args + size >= TOTAL_FLOAT_ARGS" << std::endl;
        abort();
    }
    float* ret = reinterpret_cast<float*>(bufferFloatArgs->contents()) + cur_float_args;
    cur_float_args += size;
    return ret;
}

#endif // METAL_GPU
#endif // GCC_CPU