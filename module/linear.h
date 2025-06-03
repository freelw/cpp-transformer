#ifndef LINEAR_H
#define LINEAR_H

#include "optimizers/parameter.h"

enum ACTIVATION {
    RELU,
    SIGMOID,
    TANH,
    NONE
};

class Linear {
public:
    Linear(
        int _input_num, int _output_num,
        const std::string& prefix = "",
        float w_sigma = -0.1f,
        float b_sigma = -0.1f,
        ACTIVATION act = ACTIVATION::NONE,
        bool _bias = true,
        bool const_weight = false
    );
    ~Linear() = default;
    graph::Node* forward(graph::Node* input);
    std::vector<Parameter*> get_parameters();
private:
    bool bias;
    int input_num;
    int output_num;
    graph::Node* w;
    graph::Node* b;
    Parameter* Pw;
    Parameter* Pb;
};

class LazyLinear {
public:
    LazyLinear(
        int _output_num,
        const std::string& _prefix = "",
        float _w_sigma = -0.1f,
        float _b_sigma = -0.1f,
        ACTIVATION _act = ACTIVATION::NONE,
        bool _bias = true,
        bool _const_weight = false
    ) :
        output_num(_output_num),
        prefix(_prefix),
        w_sigma(_w_sigma),
        b_sigma(_b_sigma),
        act(_act),
        bias(_bias),
        const_weight(_const_weight) {
        linear = nullptr;
    }
    ~LazyLinear();
    graph::Node* forward(graph::Node* input);
    std::vector<Parameter*> get_parameters();
private:
    Linear* linear;
    int output_num;
    std::string prefix;
    float w_sigma;
    float b_sigma;
    ACTIVATION act;
    bool bias;
    bool const_weight;
};

#endif