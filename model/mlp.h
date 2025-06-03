#ifndef V5_MLP_H
#define V5_MLP_H

#include "module/dropout.h"
#include "optimizers/parameter.h"
#include "module/linear.h"

class MLP {
public:
    MLP(
        int32_t _input,
        const std::vector<int32_t>& _outputs,
        float dropout_p,
        bool const_weight = false
    );
    ~MLP();
    std::vector<Parameter*> get_parameters();
    graph::Node* forward(graph::Node* input);
private:
    LazyLinear* l1;
    LazyLinear* l2;
    Dropout* dropout;
};

#endif
