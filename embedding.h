#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {


class Embedding {
    public:
        Embedding(uint vocab_size, uint hidden_num);
        ~Embedding();
        std::vector<Node *> forward(const std::vector<std::vector<uint>> &inputs);
        std::vector<Parameters *> get_parameters();
    private:
        uint vocab_size;
        uint hidden_num;
        std::vector<Matrix *> mW;
        std::vector<Node *> W;
        std::vector<Parameters *> PW;
};

} // namespace autograd

#endif