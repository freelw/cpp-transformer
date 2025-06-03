#ifndef ADDNORM_H
#define ADDNORM_H

#include "graph/node.h"
#include "optimizers/parameter.h"
#include "module/dropout.h"
#include "module/layernorm.h"

class AddNorm {
public:
    AddNorm(int len, float p);
    ~AddNorm();
    graph::Node* forward(graph::Node* x, graph::Node* y);
    std::vector<Parameter*> get_parameters();
private:
    LayerNorm* layer_norm;
    Dropout* dropout;
};

#endif