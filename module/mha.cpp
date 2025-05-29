#include "mha.h"

MHA::MHA(
    int num_hiddens,
    int _num_heads,
    float dropout,
    bool bias,
    bool const_weight
) : num_heads(_num_heads) {
    attention = new DotProductAttention(dropout);
    w_q = new LazyLinear(
        num_hiddens,
        "w_q",
        -1.0f, // 用LazyLinear默认初始化参数
        -1.0f, // 用LazyLinear默认初始化参数
        NONE,
        bias,
        const_weight
    );
    w_k = new LazyLinear(
        num_hiddens,
        "w_k",
        -1.0f, // 用LazyLinear默认初始化参数
        -1.0f, // 用LazyLinear默认初始化参数
        NONE,
        bias,
        const_weight
    );
    w_v = new LazyLinear(
        num_hiddens,
        "w_v",
        -1.0f, // 用LazyLinear默认初始化参数
        -1.0f, // 用LazyLinear默认初始化参数
        NONE,
        bias,
        const_weight
    );
    w_o = new LazyLinear(
        num_hiddens,
        "w_o",
        -1.0f, // 用LazyLinear默认初始化参数
        -1.0f, // 用LazyLinear默认初始化参数
        NONE,
        bias,
        const_weight
    );
}

MHA::~MHA() {
    assert(w_q != nullptr);
    assert(w_k != nullptr);
    assert(w_v != nullptr);
    assert(w_o != nullptr);
    assert(attention != nullptr);
    delete w_q;
    delete w_k;
    delete w_v;
    delete w_o;
    delete attention;
}

graph::Node *MHA::forward(
    graph::Node *queries,
    graph::Node *keys,
    graph::Node *values,
    Tensor *valid_lens
) {
    assert(queries->get_tensor()->get_dim() == 3); // shape : (batch_size, seq_len, num_hiddens)
    assert(keys->get_tensor()->get_dim() == 3);
    assert(values->get_tensor()->get_dim() == 3);
    queries = transpose_qkv(w_q->forward(queries));
    keys = transpose_qkv(w_k->forward(keys));
    values = transpose_qkv(w_v->forward(values));

    if (valid_lens != nullptr) {
        valid_lens = valid_lens->repeat_interleave(num_heads);
    }
    auto output = attention->forward(
        queries,
        keys,
        values,
        valid_lens
    );

    auto output_concat = transpose_output(output);
    return w_o->forward(output_concat);
}

std::vector<Parameter *> MHA::get_parameters() {
    std::vector<Parameter *> params;
    auto w_q_params = w_q->get_parameters();
    auto w_k_params = w_k->get_parameters();
    auto w_v_params = w_v->get_parameters();
    auto w_o_params = w_o->get_parameters();
    params.insert(params.end(), w_q_params.begin(), w_q_params.end());
    params.insert(params.end(), w_k_params.begin(), w_k_params.end());
    params.insert(params.end(), w_v_params.begin(), w_v_params.end());
    params.insert(params.end(), w_o_params.begin(), w_o_params.end());
    return params;
}

graph::Node *MHA::transpose_qkv(
    graph::Node *X
) {
    auto shape = X->get_tensor()->get_shape();
    X = X->reshape({shape[0], shape[1], num_heads, -1});
    X = X->permute({0, 2, 1, 3});
    auto shape1 = X->get_tensor()->get_shape();
    return X->reshape({-1, shape1[2], shape1[3]});
}

graph::Node *MHA::transpose_output(
    graph::Node *X
) {
    X = X->reshape({-1, num_heads, X->get_tensor()->get_shape()[1], X->get_tensor()->get_shape()[2]});
    X = X->permute({0, 2, 1, 3});
    auto shape = X->get_tensor()->get_shape();
    return X->reshape({shape[0], shape[1], -1});
}