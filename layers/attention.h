#ifndef ATTENTION_H
#define ATTENTION_H

#include "autograd/node.h"
#include "dropout.h"
#include "linear.h"


class DotProductAttetion {
    public:
        DotProductAttetion(DATATYPE _dropout);
        ~DotProductAttetion();
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &Q,
            const std::vector<autograd::Node *> &K,
            const std::vector<autograd::Node *> &V,
            const std::vector<uint> &valid_lens
        );
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &Q,
            const std::vector<autograd::Node *> &K,
            const std::vector<autograd::Node *> &V,
            const std::vector<std::vector<uint>> &valid_lens
        );
        void train(bool _is_training) {
            is_training = _is_training;
        }
        bool training() {
            return is_training;
        }
    private:
        DATATYPE dropout;
        autograd::Dropout *dropout_layer;
        bool is_training;
};

class MultiHeadAttention {
    public:
        MultiHeadAttention(uint _num_heads, uint _num_hidden, DATATYPE dropout, bool _bias = false);
        ~MultiHeadAttention();
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &queries,
            const std::vector<autograd::Node *> &keys,
            const std::vector<autograd::Node *> &values,
            const std::vector<uint> &valid_lens
        );
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &queries,
            const std::vector<autograd::Node *> &keys,
            const std::vector<autograd::Node *> &values,
            const std::vector<std::vector<uint>> &valid_lens
        );
        void train(bool _is_training) {
            is_training = _is_training;
            attention->train(_is_training);
        }
        bool training() {
            return is_training;
        }
        std::vector<autograd::Parameters *> get_parameters();
    private:
        uint num_heads;
        uint num_hidden;
        DotProductAttetion *attention;
        autograd::LazyLinear *Wq;
        autograd::LazyLinear *Wk;
        autograd::LazyLinear *Wv;
        autograd::LazyLinear *Wo;
        bool is_training;
        bool bias;
};

#endif