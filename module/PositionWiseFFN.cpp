#include "PositionWiseFFN.h"

PositionWiseFFN::PositionWiseFFN(int ffn_num_hiddens, int ffn_num_outputs) {
    dense1 = new LazyLinear(ffn_num_hiddens, "ffn_dense1", -0.1f, -0.1f, RELU);
    dense2 = new LazyLinear(ffn_num_outputs, "ffn_dense2");
}

PositionWiseFFN::~PositionWiseFFN() {
    delete dense1;
    delete dense2;
}

graph::Node* PositionWiseFFN::forward(graph::Node* x) {
    x = dense1->forward(x);
    x = x->relu();
    x = dense2->forward(x);
    return x;
}

std::vector<Parameter*> PositionWiseFFN::get_parameters() {
    std::vector<Parameter*> params;
    auto dense1_params = dense1->get_parameters();
    auto dense2_params = dense2->get_parameters();
    params.insert(params.end(), dense1_params.begin(), dense1_params.end());
    params.insert(params.end(), dense2_params.begin(), dense2_params.end());
    return params;
}