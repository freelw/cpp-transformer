#include "stats/stats.h"
#include "matrix/matrix.h"
#include "autograd/node.h"

namespace autograd {
    std::string stats() {
        std::string res;

        TmpMatricsStats matrics_stats = tmpMatricsStats();
        TmpNodesStats nodes_stats = tmpNodesStats();
        TmpEdgesStats edges_stats = tmpEdgesStats();

        res += "TmpMatricsStats : ";
        res += "size : " + std::to_string(matrics_stats.size) + ", ";
        res += "bytes : " + std::to_string(matrics_stats.bytes) + "\n";
        res += "TmpNodesStats : ";
        res += "size : " + std::to_string(nodes_stats.size) + ", ";
        res += "bytes : " + std::to_string(nodes_stats.bytes) + "\n";
        res += "TmpEdgesStats : ";
        res += "size : " + std::to_string(edges_stats.size) + ", ";
        res += "bytes : " + std::to_string(edges_stats.bytes) + "\n";
        return res;
    }
}
