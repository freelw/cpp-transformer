#ifndef METAL_OPS_H
#define METAL_OPS_H

#include "backends/backend_ops.h"

#ifndef GCC_CPU
#ifdef METAL_GPU
#include <Metal/Metal.h>
class MetalOps : public BackendOps {
};

#endif // METAL_GPU
#endif // GCC_CPU

#endif