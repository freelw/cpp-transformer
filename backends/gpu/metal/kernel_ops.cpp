#include "kernel_ops.h"

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <vector>
#include <type_traits>

MetalKops::MetalKops(const std::string& functionName, MTL::Library* library)
    : functionName(functionName), commandBuffer(nullptr), encoder(nullptr) {
    function = library->newFunction(NS::String::string(functionName.c_str(), NS::StringEncoding::UTF8StringEncoding));
    if (!function) {
        std::cerr << "Error: Function '" << functionName << "' not found in Metal library." << std::endl;
        throw std::runtime_error("Function not found in Metal library");
    }
}

MetalKops::~MetalKops() {
    function->release();
}

void MetalKops::prepare(MTL::Device* device, MTL::CommandQueue* commandQueue, NS::Error* error) {
    commandBuffer = commandQueue->commandBuffer();
    encoder = commandBuffer->computeCommandEncoder();
    if (!encoder) {
        std::cerr << "Error: Failed to create compute command encoder." << std::endl;
        throw std::runtime_error("Failed to create compute command encoder");
    }
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(function, &error);
    if (!pipelineState) {
        std::cerr << "Error creating compute pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        throw std::runtime_error("Failed to create compute pipeline state");
    }
    encoder->setComputePipelineState(pipelineState);
}

void MetalKops::run() {

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    encoder->release();
    commandBuffer->release();
}