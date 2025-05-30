#include "layernorm.h"

LayerNorm::LayerNorm(int len, bool const_weight) {
    Tensor *t_gamma = allocTensor({len}, "layernorm_gamma"); // do not calloc
    Tensor *t_beta = allocTensor({len}, "layernorm_beta"); // do not calloc
    gamma = graph::allocNode(t_gamma);
    beta = graph::allocNode(t_beta);
    gamma->require_grad();
    beta->require_grad();
    gamma->init_weight_fill(1.0);
    beta->init_weight_fill(0.0);    
    Pgamma = allocParameter(gamma);
    Pbeta = allocParameter(beta);
}

graph::Node* LayerNorm::forward(graph::Node *x) {
    assert(x->get_tensor()->get_dim() >= 2);
    auto origin_shape = x->get_tensor()->get_shape();
    auto dim = x->get_tensor()->get_dim();
    x = x->reshape({-1, origin_shape[dim - 1]});
    x = x->norm()->expand_mul(gamma)->expand_add(beta);
    return x->reshape(origin_shape);
}

std::vector<Parameter*> LayerNorm::get_parameters() {
    std::vector<Parameter*> params;
    params.push_back(Pgamma);
    params.push_back(Pbeta);
    return params;
}