#ifndef LAYERNORM_H
#define LAYERNORM_H


#include "graph/node.h"
#include "optimizers/parameter.h"
class LayerNorm {
    public:
        LayerNorm(int len, bool const_weight = false);
        ~LayerNorm() = default;
        graph::Node* forward(graph::Node *x);
        std::vector<Parameter*> get_parameters();
    private:
        graph::Node *gamma;
        graph::Node *beta;
        Parameter *Pgamma;
        Parameter *Pbeta;
};
#endif