#ifndef PARAMETER_H
#define PARAMETER_H

#include "graph/node.h"

struct ParameterInfo {
    int weight_size;
    int m_size;
    int v_size;
    int t;
    char* weight_start;
    char* m_start;
    char* v_start;
};

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
    void deserialize_info(char* buffer, ParameterInfo* info);
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