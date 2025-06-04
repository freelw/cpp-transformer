#ifndef POSITIONWISEFFN_H
#define POSITIONWISEFFN_H

#include "graph/node.h"
#include "optimizers/parameter.h"
#include "module/linear.h"

class PositionWiseFFN {
public:
    PositionWiseFFN(int ffn_num_hiddens, int ffn_num_outputs);
    ~PositionWiseFFN();
    graph::Node* forward(graph::Node* x);
    std::vector<Parameter*> get_parameters();
private:
    LazyLinear* dense1;
    LazyLinear* dense2;
};

#endif