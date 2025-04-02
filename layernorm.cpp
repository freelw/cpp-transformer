#include "layernorm.h"

LayerNorm::LayerNorm(uint dim) {
    mgamma = new Matrix(Shape(dim, 1));
    mbeta = new Matrix(Shape(dim, 1));
    // mgamma->fill(1);
    // mbeta->fill(0);
    init_weight(mgamma, 0.01, 1);
    init_weight(mbeta, 0.01, 0);
    gamma = new autograd::Node(mgamma, true);
    beta = new autograd::Node(mbeta, true);
    gamma->require_grad();
    beta->require_grad();
    pgamma = new autograd::Parameters(gamma);
    pbeta = new autograd::Parameters(beta);
}

LayerNorm::~LayerNorm() {
    delete mgamma;
    delete mbeta;
    delete gamma;
    delete beta;
    delete pgamma;
    delete pbeta;
}

autograd::Node* LayerNorm::forward(autograd::Node* x) {
    std::vector<autograd::Node*> v_gamma;
    for (uint i = 0; i < x->get_weight()->getShape().colCnt; i++) {
        v_gamma.push_back(gamma);
    }
    autograd::Node *gammas = autograd::cat(v_gamma, 0);
    return (*gammas * x->Norm())->expand_add(beta);
}

std::vector<autograd::Parameters*> LayerNorm::parameters() {
    std::vector<autograd::Parameters*> res;
    res.push_back(pgamma);
    res.push_back(pbeta);
    return res;
}