#include "backends/backend_ops.h"
#include "parameter.h"

#include <cstring>

Parameter::Parameter(graph::Node *_node)
    : node(_node), t(0) {
    Tensor *t = _node->get_tensor();
    m = allocTensor(t->get_shape(), t->get_name()+"_m"); // do not calloc
    v = allocTensor(t->get_shape(), t->get_name()+"_v"); // do not calloc
}

Tensor *Parameter::get_w() {
    return node->get_tensor();
}

Tensor *Parameter::get_grad() {
    assert(node->is_require_grad());
    assert(node->get_grad() != nullptr);
    return node->get_grad();
}

Tensor *Parameter::get_m() {
    return m;
}

Tensor *Parameter::get_v() {
    return v;
}

bool Parameter::is_require_grad() {
    return node->is_require_grad();
}

std::string Parameter::serialize() {
    int weight_size = get_w()->size();
    int grad_size = get_grad()->size();
    int m_size = m->size();
    int v_size = v->size();

    int tot_size = 0;
    tot_size += sizeof(weight_size);
    tot_size += sizeof(grad_size);
    tot_size += sizeof(m_size);
    tot_size += sizeof(v_size);
    tot_size += sizeof(t);
    tot_size += weight_size;
    tot_size += grad_size;
    tot_size += m_size;
    tot_size += v_size;
    
    char *buffer = static_cast<char *>(::malloc(tot_size)); 
    int offset = 0;

    ::memcpy(buffer + offset, &weight_size, sizeof(weight_size));
    offset += sizeof(weight_size);
    ::memcpy(buffer + offset, &grad_size, sizeof(grad_size));
    offset += sizeof(grad_size);
    ::memcpy(buffer + offset, &m_size, sizeof(m_size));
    offset += sizeof(m_size);
    ::memcpy(buffer + offset, &v_size, sizeof(v_size));
    offset += sizeof(v_size);
    ::memcpy(buffer + offset, &t, sizeof(t));
    offset += sizeof(t);

    g_backend_ops->cp_from_device(
        buffer + offset,
        get_w(),
        weight_size
    );
    offset += weight_size;
    g_backend_ops->cp_from_device(
        buffer + offset,
        get_grad(),
        grad_size
    );
    offset += grad_size;
    g_backend_ops->cp_from_device(
        buffer + offset,
        m,
        m_size
    );
    offset += m_size;
    g_backend_ops->cp_from_device(
        buffer + offset,
        v,
        v_size
    );
    offset += v_size;
    assert(offset == tot_size);
    std::string res((char *)buffer, tot_size);
    ::free(buffer);
    return res;
}

void Parameter::deserialize(char *buffer) {
    int weight_size, grad_size, m_size, v_size;
    int offset = 0;

    ::memcpy(&weight_size, buffer + offset, sizeof(weight_size));
    offset += sizeof(weight_size);
    ::memcpy(&grad_size, buffer + offset, sizeof(grad_size));
    offset += sizeof(grad_size);
    ::memcpy(&m_size, buffer + offset, sizeof(m_size));
    offset += sizeof(m_size);
    ::memcpy(&v_size, buffer + offset, sizeof(v_size));
    offset += sizeof(v_size);
    ::memcpy(&t, buffer + offset, sizeof(t));
    offset += sizeof(t);

    assert(weight_size == get_w()->size());
    assert(grad_size == get_grad()->size());
    assert(m_size == m->size());
    assert(v_size == v->size());

    g_backend_ops->cp_to_device(
        get_w(),
        buffer + offset,
        weight_size
    );
    offset += weight_size;
    g_backend_ops->cp_to_device(
        get_grad(),
        buffer + offset,
        grad_size
    );
    offset += grad_size;
    g_backend_ops->cp_to_device(
        m,
        buffer + offset,
        m_size
    );
    offset += m_size;
    g_backend_ops->cp_to_device(
        v,
        buffer + offset,
        v_size
    );
    offset += v_size;
    assert(offset == get_serialized_size());
}

int Parameter::get_serialized_size() {
    return sizeof(int) * 4 + sizeof(t) + 
           get_w()->size() + get_grad()->size() + m->size() + v->size();
}

std::vector<Parameter *> g_parameters;

Parameter *allocParameter(graph::Node *_node) {
    Parameter *param = new Parameter(_node);
    g_parameters.push_back(param);
    return param;
}

void releaseParameters() {
    for (auto param : g_parameters) {
        delete param;
    }
    g_parameters.clear();
}