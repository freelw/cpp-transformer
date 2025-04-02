#include "ffn.h"

PositionwiseFFN::PositionwiseFFN(uint _num_hidden, uint _num_out)
    : num_hidden(_num_hidden), num_out(_num_out) {
    dense1 = new autograd::LazyLinear(num_hidden, autograd::ACTIVATION::RELU, true);
    dense2 = new autograd::LazyLinear(num_out,  autograd::ACTIVATION::NONE, true);
    // #pragma message("dense bias should be random, fix later.")
}

PositionwiseFFN::~PositionwiseFFN() {
    delete dense1;
    delete dense2;
}

autograd::Node *PositionwiseFFN::forward(autograd::Node *x) {
    return dense2->forward(dense1->forward(x)->Relu());
}

std::vector<autograd::Node *> PositionwiseFFN::forward(const std::vector<autograd::Node *> &x) {
    std::vector<autograd::Node *> res;
    for (auto & _x : x) {
        res.push_back(forward(_x));
    }
    return res;
}

std::vector<autograd::Parameters *> PositionwiseFFN::get_parameters() {
    std::vector<autograd::Parameters *> res;
    auto p1 = dense1->get_parameters();
    auto p2 = dense2->get_parameters();
    res.insert(res.end(), p1.begin(), p1.end());
    res.insert(res.end(), p2.begin(), p2.end());
    return res;
}