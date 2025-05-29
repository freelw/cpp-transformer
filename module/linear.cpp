#include "linear.h"
#include <cmath>

Linear::Linear(
    int _input_num, int _output_num,
    const std::string & prefix,
    float w_sigma,
    float b_sigma,
    ACTIVATION act,
    bool _bias,
    bool const_weight
): input_num(_input_num), output_num(_output_num), bias(_bias){
    auto w_tensor = allocTensor({input_num, output_num}, prefix + "_w_linear"); // do not calloc
    w = graph::allocNode(w_tensor);
    w->require_grad();
    Pw = allocParameter(w);

    float mean = 0.0f;

    if (const_weight) {
        w->init_weight_for_dbg();
        return;
    } else {
        if (w_sigma <  0) {
            switch (act) {
                case RELU:
                    w_sigma = std::sqrt(2.0f / input_num);
                    break;
                case SIGMOID:
                case NONE:
                    w_sigma = std::sqrt(2.0f / (input_num + output_num));
                    break;
                case TANH:
                    w_sigma = std::sqrt(2.0 / (input_num + output_num)) * 4;
                    break;
                default:
                    assert(false);
            }
        }
        w->init_weight_gauss(w_sigma, mean);
    }
    if (bias) {
        auto b_tensor = allocTensor({output_num}, prefix + "_b_linear"); // do not calloc
        b = graph::allocNode(b_tensor);
        b->require_grad();
        if (!const_weight) {
            if (b_sigma >= 0) {
                b->init_weight_gauss(b_sigma, b_sigma);

            } else {
                b->init_weight_gauss(0.01, 0.01); // this is very important, break the symmetry
            }
        }
        Pb = allocParameter(b);
    } else {
        b = nullptr;
    }
}

graph::Node *Linear::forward(graph::Node *input) {
    auto dim = input->get_tensor()->get_dim();
    auto input_shape = input->get_tensor()->get_shape();
    assert(dim >= 2);
    if (dim > 2) {
        input = input->reshape({-1, input_num});
    }
    auto res = input->at(w);
    if (bias) {
        assert(b != nullptr);
        res = res->expand_add(b);
    }
    if (dim > 2) {
        auto shape = input_shape;
        shape.pop_back();
        shape.push_back(output_num);
        res = res->reshape(shape);
    }
    return res;
}

std::vector<Parameter *> Linear::get_parameters() {
    std::vector<Parameter *> params;
    params.push_back(Pw);
    if (bias) {
        assert(b != nullptr);
        assert(Pb != nullptr);
        params.push_back(Pb);
    }
    return params;
}

LazyLinear::~LazyLinear() {
    if (linear != nullptr) {
        delete linear;
    }
}

graph::Node *LazyLinear::forward(graph::Node *input) {
    if (linear == nullptr) {
        auto shape = input->get_tensor()->get_shape();
        auto dim = input->get_tensor()->get_dim();
        auto input_num = shape[dim-1];
        linear = new Linear(
            input_num, output_num,
            prefix, w_sigma, b_sigma,
            act, bias, const_weight
        );
    }
    return linear->forward(input);
}

std::vector<Parameter *> LazyLinear::get_parameters() {
    assert(linear != nullptr);
    return linear->get_parameters();
}