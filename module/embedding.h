#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "graph/node.h"
#include "optimizers/parameter.h"

class Embedding {
    public:
        Embedding(int _vocab_size, int _hidden_num, bool const_weight = false);
        ~Embedding() = default;
        graph::Node *forward(Tensor *indices);
        std::vector<Parameter *> get_parameters();
        Tensor *get_weight() {
            return w->get_tensor();
        }
        Tensor *get_grad() {
            return w->get_grad();
        }
    private:
        int vocab_size;
        int hidden_num;
        graph::Node *w;
        Parameter *Pw;
};

#endif