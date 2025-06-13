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
    void prepare(MTL::Device* device, MTL::CommandQueue* commandQueue);
    void run();
    MTL::ComputeCommandEncoder* getEncoder() const {
        return encoder;
    }
private:
    std::string functionName;
    MTL::Function* function;
    MTL::CommandBuffer* commandBuffer;
    MTL::ComputeCommandEncoder* encoder;
    MTL::ComputePipelineState* pipelineState;
};


#endif