#include "kernel_ops.h"

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <vector>
#include <type_traits>

MetalKops::MetalKops(const std::string& functionName, MTL::Library* library)
    : functionName(functionName) {
    function = library->newFunction(NS::String::string(functionName.c_str(), NS::StringEncoding::UTF8StringEncoding));
    if (!function) {
        std::cerr << "Error: Function '" << functionName << "' not found in Metal library." << std::endl;
        throw std::runtime_error("Function not found in Metal library");
    }
}

MetalKops::~MetalKops() {
    function->release();
}