#include "embedding.h"
#include <cmath>

Embedding::Embedding(int _vocab_size, int _hidden_num, bool const_weight)
    : vocab_size(_vocab_size), hidden_num(_hidden_num) {
    Tensor* t = allocTensor({ vocab_size, hidden_num }, "embedding"); // do not calloc
    w = graph::allocNode(t);
    w->require_grad();
    if (const_weight) {
        w->init_weight_for_dbg(10000.0f);
    } else {
        w->init_weight_uniform(std::sqrt(1.0 / hidden_num));
    }
    Pw = allocParameter(w);
}

graph::Node* Embedding::forward(Tensor* indices) {
    assert(indices->get_dtype() == INT32);
    assert(indices->get_dim() == 2);
    assert(indices->is_contiguous());
    auto origin_shape = indices->get_shape();
    indices = indices->reshape({ -1 });
    Tensor* res = callocTensor({ indices->get_shape()[0], hidden_num }, "embedding_out");
    auto res_node = graph::allocNode(res);
    res_node->require_grad();
    gCreateAction(
        new EmbeddingAction(
            w->get_tensor(),
            indices,
            res
        )
    );
    res_node->edges.push_back(
        graph::EmbeddingEdge::create(w, indices)
    );
    return res_node->reshape({ origin_shape[0], origin_shape[1], hidden_num });
}

std::vector<Parameter*> Embedding::get_parameters() {
    return { Pw };
}