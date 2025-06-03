#include "mlp.h"
#include "graph/node.h"

MLP::MLP(
    int32_t _input,
    const std::vector<int32_t>& _outputs,
    float dropout_p,
    bool const_weight
) {
    auto w_sigma = 0.02f;
    auto b_sigma = -0.01f; // set bias 0
    l1 = new LazyLinear(_outputs[0], "l1", w_sigma, b_sigma, RELU, true, const_weight);
    l2 = new LazyLinear(_outputs[1], "l2", w_sigma, b_sigma, NONE, true, const_weight);
    dropout = new Dropout(dropout_p);
}

MLP::~MLP() {
    assert(l1 != nullptr);
    assert(l2 != nullptr);
    assert(dropout != nullptr);
    delete l1;
    delete l2;
    delete dropout;
}

std::vector<Parameter*> MLP::get_parameters() {
    std::vector<Parameter*> params;
    auto l1_params = l1->get_parameters();
    auto l2_params = l2->get_parameters();
    params.insert(params.end(), l1_params.begin(), l1_params.end());
    params.insert(params.end(), l2_params.begin(), l2_params.end());
    return params;
}

graph::Node* MLP::forward(graph::Node* input) {
    auto x = l1->forward(input);
    x = x->relu();
    x = l2->forward(x);
    auto x_shape = x->get_tensor()->get_shape();
    x = dropout->forward(x)->reshape(x_shape);
    return x;
}