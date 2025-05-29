#ifndef ATTENTION_H
#define ATTENTION_H

#include "graph/node.h"
#include "dropout.h"

class DotProductAttention {
    public:
        DotProductAttention(float p = 0.0f);
        ~DotProductAttention();
        graph::Node *forward(
            graph::Node *query, graph::Node *key,
            graph::Node *value, Tensor *valid_lens
        );
    public:
        graph::Node *attention_weights;
        Dropout *dropout;
};

#endif