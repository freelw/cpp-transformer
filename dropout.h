#ifndef DROPOUT_H
#define DROPOUT_H

#include "autograd/node.h"
#include <random>
#include <chrono>

namespace autograd {

    extern bool dropout_run;

    class Dropout {
        public:
            Dropout(DATATYPE _dropout);
            ~Dropout() {}
            std::vector<Node *> forward(const std::vector<Node *> &inputs);
            Node *forward(Node *input);
        private:
            DATATYPE dropout;
            std::mt19937 gen;
            std::uniform_real_distribution<> dis;
    };
} // namespace autograd
#endif