#include "addnorm.h"

AddNorm::AddNorm(int len, float p) {
    layer_norm = new LayerNorm(len);
    dropout = new Dropout(p);
}

AddNorm::~AddNorm() {
    delete layer_norm;
    delete dropout;
}

graph::Node* AddNorm::forward(graph::Node* x, graph::Node* y) {
    auto drop_y = dropout->forward(y);
    auto sum = drop_y->add(x);
    return layer_norm->forward(sum);
}

std::vector<Parameter*> AddNorm::get_parameters() {
    return layer_norm->get_parameters();
}