#ifndef PARAMETER_H
#define PARAMETER_H

#include "graph/node.h"

class Parameter {
public:
    Parameter(graph::Node* _node);
    Tensor* get_w();
    Tensor* get_grad();
    Tensor* get_m();
    Tensor* get_v();
    bool is_require_grad();
    void inc_t() {
        t++;
    }
    int get_t() {
        return t;
    }
    std::string serialize();
    void deserialize(char* buffer);
    int get_serialized_size();

private:
    graph::Node* node;
    Tensor* m;
    Tensor* v;
    int t;
};

Parameter* allocParameter(graph::Node* _node);
void releaseParameters();
#endif