#ifndef DROPOUT_H
#define DROPOUT_H

#include "graph/node.h"

class Dropout {
    public:
        Dropout(float _p) : p(_p) {}
        graph::Node *forward(graph::Node *x);
    private:
        float p;
};


#endif