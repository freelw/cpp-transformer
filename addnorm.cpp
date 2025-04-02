#include "addnorm.h"

AddNorm::AddNorm(int _num_hidden, DATATYPE _dropout)
    : num_hidden(_num_hidden),
    training(true),
    dropout(_dropout),
    dropout_layer(nullptr),
    layernorm(nullptr)
{
    if (dropout > 0) {
        dropout_layer = new autograd::Dropout(dropout);
    }
    layernorm = new LayerNorm(num_hidden);
}

AddNorm::~AddNorm() {
    if (dropout > 0 && dropout_layer != nullptr) {
        delete dropout_layer;
    }
    delete layernorm;
}

autograd::Node *AddNorm::forward(autograd::Node *x, autograd::Node *y) {
    assert(x->checkShape(y->getShape()));
    autograd::Node * _y;
    if (dropout > 0 && is_training()) {
        _y = dropout_layer->forward(y);
    } else {
        _y = y;
    }
    return layernorm->forward((*x) + _y);
}

std::vector<autograd::Parameters *> AddNorm::get_parameters() {
    return layernorm->parameters();
}

std::vector<autograd::Node *> AddNorm::forward(
    const std::vector<autograd::Node *> &x,
    const std::vector<autograd::Node *> &y) {
    assert(x.size() == y.size());
    std::vector<autograd::Node *> res;
    for (size_t i = 0; i < x.size(); i++) {
        res.push_back(forward(x[i], y[i]));
    }
    return res;
}
