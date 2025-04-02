#ifndef POSITIONWISE_FFN_H
#define POSITIONWISE_FFN_H

#include "autograd/node.h"
#include "dropout.h"
#include "linear.h"

class PositionwiseFFN {
    public:
        PositionwiseFFN(uint _num_hidden, uint _num_out);
        ~PositionwiseFFN();
        autograd::Node *forward(autograd::Node *x);
        std::vector<autograd::Node *> forward(const std::vector<autograd::Node *> &x);
        std::vector<autograd::Parameters *> get_parameters();
    private:
        uint num_hidden;
        uint num_out;
        autograd::LazyLinear *dense1;
        autograd::LazyLinear *dense2;
};

#endif