#ifndef AUTOGRAD_STATS_H
#define AUTOGRAD_STATS_H

#include <iostream>
namespace autograd {
    struct TmpMatricsStats {
        unsigned long long size;
        unsigned long long bytes;
    };
    struct TmpNodesStats {
        unsigned long long size;
        unsigned long long bytes;
    };
    struct TmpEdgesStats {
        unsigned long long size;
        unsigned long long bytes;
    };
    std::string stats();
} // namespace autograd
#endif