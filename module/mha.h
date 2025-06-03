#ifndef MHA_H
#define MHA_H
#include "attention.h"
#include "linear.h"

class MHA {
public:
    MHA(
        int num_hiddens,
        int _num_heads,
        float dropout = 0.0f,
        bool bias = false,
        bool const_weight = false
    );
    ~MHA();
    graph::Node* forward(
        graph::Node* queries,
        graph::Node* keys,
        graph::Node* values,
        Tensor* valid_lens = nullptr
    );
    std::vector<Parameter*> get_parameters();

private:
    graph::Node* transpose_qkv(
        graph::Node* X
    );
    graph::Node* transpose_output(
        graph::Node* X
    );

private:
    int num_heads;
    DotProductAttention* attention;
    LazyLinear* w_q;
    LazyLinear* w_k;
    LazyLinear* w_v;
    LazyLinear* w_o;
};

#endif