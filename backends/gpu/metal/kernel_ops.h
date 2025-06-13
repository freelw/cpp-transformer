#ifndef KERNELOPS_H
#define KERNELOPS_H

#include <string>

namespace MTL {
    class Library;
    class Function;
}

class MetalKops {
public:
    MetalKops(const std::string& functionName, MTL::Library* library);
    virtual ~MetalKops();
    const std::string& getFunctionName() const {
        return functionName;
    }
private:
    std::string functionName;
    MTL::Function* function;
};


#endif