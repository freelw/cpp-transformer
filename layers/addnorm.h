#ifndef ADDNORM_H
#define ADDNORM_H

#include "autograd/node.h"
#include "dropout.h"
#include "layernorm.h"

class AddNorm {
    public:
        AddNorm(int _num_hidden, DATATYPE _dropout);
        ~AddNorm();
        autograd::Node *forward(autograd::Node *x, autograd::Node *y);
        std::vector<autograd::Node *> forward(const std::vector<autograd::Node *> &x, const std::vector<autograd::Node *> &y);
        std::vector<autograd::Parameters *> get_parameters();
        void train(bool _training) { training = _training; }
        bool is_training() { return training; }
    private:
        int num_hidden;
        bool training;
        DATATYPE dropout;
        autograd::Dropout *dropout_layer;
        LayerNorm *layernorm;
};

#endif