#ifndef GCC_CPU
#ifdef METAL_GPU

#ifndef KERNELOPS_H
#define KERNELOPS_H

#include <string>

namespace MTL {
    class Library;
    class Function;
    class Device;
    class CommandQueue;
    class CommandBuffer;
    class ComputeCommandEncoder;
    class ComputePipelineState;
}

namespace NS {
    class Error;
}

class MetalKops {
public:
    MetalKops(const std::string& functionName, MTL::Library* library);
    virtual ~MetalKops();
    const std::string& getFunctionName() const {
        return functionName;
    }
    MTL::ComputeCommandEncoder* prepare(
        MTL::Device* device,
        MTL::CommandQueue* commandQueue,
        MTL::CommandBuffer* commandBuffer
    );
private:
    std::string functionName;
    MTL::Function* function;
};


#endif // KERNELOPS_H
#endif // METAL_GPU
#endif // GCC_CPU
