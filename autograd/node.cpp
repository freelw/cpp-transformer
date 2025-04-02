#include "node.h"
#include <assert.h>
#include <iostream>
#include <string.h>
#include "stats/stats.h"

namespace autograd {

    Matrix *CrossEntropyLoss(Matrix *input, const std::vector<uint> &labels, std::vector<CrosEntropyInfo> &info) { 
        assert(input->getShape().colCnt == labels.size());
        assert(info.size() == 0);
        Matrix *loss = allocTmpMatrix(Shape(1,1));
        DATATYPE loss_value = 0;
        info.resize(input->getShape().colCnt);

        #pragma omp parallel for reduction(+:loss_value)
        for (uint j = 0; j < input->getShape().colCnt; ++ j) {
            DATATYPE max = (*input)[0][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                auto e = (*input)[i][j];
                if (max < e) {
                    max = e;
                }
            }
            DATATYPE sum = 0;
            auto target = labels[j];
            DATATYPE zt = (*input)[target][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                DATATYPE e = (*input)[i][j];
                e = std::exp(e-max);
                sum += e;
            }
            CrosEntropyInfo &p = info[j];
            p.sum = sum;
            p.max = max;
            loss_value += -(zt - max - log(sum));
        }
        (*loss)[0][0] = loss_value/labels.size();
        return loss;
    }

    Matrix *CrossEntropyLossMask(
        Matrix *input,
        const std::vector<uint> &labels,
        std::vector<CrosEntropyInfo> &info,
        const std::vector<bool> &mask) {

        assert(input->getShape().colCnt == labels.size());
        assert(input->getShape().colCnt == mask.size());
        assert(info.size() == 0);

        Matrix *loss = allocTmpMatrix(Shape(1,1));
        DATATYPE loss_value = 0;
        info.resize(input->getShape().colCnt);
        uint mask_cnt = 0;
        // #pragma omp parallel for reduction(+:mask_cnt)
        for (uint i = 0; i < mask.size(); ++ i) {
            mask_cnt += mask[i];
        }
        if (mask_cnt == 0) {
            (*loss)[0][0] = 0;
            return loss;
        }

        // #pragma omp parallel for reduction(+:loss_value)
        for (uint j = 0; j < input->getShape().colCnt; ++ j) {
            if (!mask[j]) {
                continue;
            }
            DATATYPE max = (*input)[0][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                auto e = (*input)[i][j];
                if (max < e) {
                    max = e;
                }
            }
            DATATYPE sum = 0;
            auto target = labels[j];
            DATATYPE zt = (*input)[target][j];
            for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
                DATATYPE e = (*input)[i][j];
                e = std::exp(e-max);
                sum += e;
            }
            CrosEntropyInfo &p = info[j];
            p.sum = sum;
            p.max = max;
            loss_value += -(zt - max - log(sum));
        }
        (*loss)[0][0] = loss_value/mask_cnt;
        return loss;
    }

    Node *Node::Norm() {
        auto *tmp = allocTmpMatrix(w);
        std::vector<DATATYPE> avg_res = w->avg();
        std::vector<DATATYPE> var_res = w->var();
        Shape shape = tmp->getShape();
        DATATYPE eps = 1e-5;
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                (*tmp)[i][j] = ((*w)[i][j] - avg_res[j]) / sqrt(var_res[j] + eps);
            }
        }
        
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(NormEdge::create(this, tmp, avg_res, var_res, eps));
        }
        return node;
    }

    Node *Node::Softmax() {
        auto *tmp = allocTmpMatrix(w);
        Shape shape = tmp->getShape();
        for (uint j = 0; j < shape.colCnt; ++ j) {
            DATATYPE max = (*w)[0][j];
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                if (max < (*w)[i][j]) {
                    max = (*w)[i][j];
                }
            }
            DATATYPE sum = 0;
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                DATATYPE e = std::exp((*w)[i][j] - max);
                sum += e;
                (*tmp)[i][j] = e;
            }
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                (*tmp)[i][j] /= sum;
            }
        }
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(SoftmaxEdge::create(this, node));
        }
        return node;
    }

    Node *Node::Transpose() {
        auto *tmp = allocTmpMatrix(w->transpose());
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(TransposeEdge::create(this));
        }
        return node;
    }

    Node *Node::Mul(DATATYPE v) {
        auto *tmp = allocTmpMatrix(*w * v);
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(MulSingleValueEdge::create(this, v));
        }
        return node;
    }

    Node *Node::Div(DATATYPE v) {
        auto *tmp = allocTmpMatrix(*w / v);
        auto *node = allocNode(tmp);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(DivEdge::create(this, v));
        }
        return node;
    }
 
    Node *Node::operator+(Node *rhs) {
        auto *node = allocNode(*w + *(rhs->w));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(AddEdge::create(rhs));
            }
        }
        return node;
    }

    Node *Node::operator*(Node *rhs) {
        auto *node = allocNode(*w * *(rhs->w));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(MulEdge::create(this, rhs->get_weight()));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(MulEdge::create(rhs, w));
            }
        }
        return node;
    }

    Node *Node::expand_add(Node *rhs) {
        auto *node = allocNode(w->expand_add(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(ExpandAddEdge::create(rhs));
            }
        }
        return node;
    }

    Node *Node::at(Node *rhs) {
        auto *node = allocNode(w->at(*(rhs->w)));
        if (is_require_grad() || rhs->is_require_grad()) {
            node->require_grad();
            if (is_require_grad()) {
                node->edges.push_back(MatMulLEdge::create(this, rhs->get_weight()));
            }
            if (rhs->is_require_grad()) {
                node->edges.push_back(MatMulREdge::create(rhs, w));
            }
        }
        return node;
    }

    Node *Node::Relu() {
        auto *node = allocNode(w->Relu());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(ReluEdge::create(this));
        }
        return node;
    }

    Node *Node::CrossEntropy(const std::vector<uint> &labels) {
        assert(w->getShape().colCnt == labels.size());
        std::vector<CrosEntropyInfo> info;
        auto *node = allocNode(::autograd::CrossEntropyLoss(w, labels, info));
        assert(info.size() == w->getShape().colCnt);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(CrossEntropyEdge::create(this, labels, info));
        }
        return node;
    }

    Node *Node::CrossEntropyMask(const std::vector<uint> &labels, const std::vector<bool> &mask) {
        assert(w->getShape().colCnt == labels.size());
        assert(w->getShape().colCnt == mask.size());
        std::vector<CrosEntropyInfo> info;
        auto *node = allocNode(::autograd::CrossEntropyLossMask(w, labels, info, mask));
        assert(info.size() == w->getShape().colCnt);
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(CrossEntropyMaskEdge::create(this, labels, mask, info));
        }
        return node;
    }

    Node *Node::Tanh() {
        auto *node = allocNode(w->tanh());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(TanhEdge::create(this));
        }
        return node;
    }

    Node *Node::Sigmoid() {
        auto *node = allocNode(w->sigmoid());
        if (is_require_grad()) {
            node->require_grad();
            node->edges.push_back(SigmoidEdge::create(this));
        }
        return node;
    }

    Node *operator-(DATATYPE v, Node &rhs) {
        auto node = allocNode(v - *(rhs.get_weight()));
        if (rhs.is_require_grad()) {
            node->require_grad();
            node->edges.push_back(MinusEdge::create(&rhs));
        }
        return node;
    }

    Node *cat0(const std::vector<Node *> &nodes) {
        Shape shape = nodes[0]->get_weight()->getShape();
        for (uint i = 0; i < nodes.size(); ++ i) {
            nodes[i]->checkShape(shape);
        }
        Matrix *m = allocTmpMatrix(Shape(shape.rowCnt, shape.colCnt * nodes.size()));
        Node *node = allocNode(m);
        
        auto m_buffer = m->getData();
        auto m_shape = m->getShape();
        
        for (uint i = 0; i < nodes.size(); ++ i) {
            auto node_i_buffer = nodes[i]->get_weight()->getData();
            
            int k = 0;

            for (; k < (int)shape.rowCnt - 7; k += 8) {

                DATATYPE *m_buffer_0 = m_buffer + k*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_1 = m_buffer + (k+1)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_2 = m_buffer + (k+2)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_3 = m_buffer + (k+3)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_4 = m_buffer + (k+4)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_5 = m_buffer + (k+5)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_6 = m_buffer + (k+6)*m_shape.colCnt+i*shape.colCnt;
                DATATYPE *m_buffer_7 = m_buffer + (k+7)*m_shape.colCnt+i*shape.colCnt;

                DATATYPE *node_i_buffer_0 = node_i_buffer + k*shape.colCnt;
                DATATYPE *node_i_buffer_1 = node_i_buffer + (k+1)*shape.colCnt;
                DATATYPE *node_i_buffer_2 = node_i_buffer + (k+2)*shape.colCnt;
                DATATYPE *node_i_buffer_3 = node_i_buffer + (k+3)*shape.colCnt;
                DATATYPE *node_i_buffer_4 = node_i_buffer + (k+4)*shape.colCnt;
                DATATYPE *node_i_buffer_5 = node_i_buffer + (k+5)*shape.colCnt;
                DATATYPE *node_i_buffer_6 = node_i_buffer + (k+6)*shape.colCnt;
                DATATYPE *node_i_buffer_7 = node_i_buffer + (k+7)*shape.colCnt;

                DATATYPE *m_buffers[8] = {
                    m_buffer_0, m_buffer_1, m_buffer_2, m_buffer_3,
                    m_buffer_4, m_buffer_5, m_buffer_6, m_buffer_7
                };

                DATATYPE *node_i_buffers[8] = {
                    node_i_buffer_0, node_i_buffer_1, node_i_buffer_2, node_i_buffer_3,
                    node_i_buffer_4, node_i_buffer_5, node_i_buffer_6, node_i_buffer_7
                };

                #pragma omp parallel for num_threads(8)
                for (uint j = 0; j < 8; ++ j) {
                    memcpy(m_buffers[j], node_i_buffers[j], shape.colCnt * sizeof(DATATYPE));
                }
            }
            for (; k < (int)shape.rowCnt; ++ k) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    m_buffer[k*m_shape.colCnt+i*shape.colCnt+j] = node_i_buffer[k*shape.colCnt+j];
                }
            }
        }
        for (uint i = 0; i < nodes.size(); ++ i) {
            if (nodes[i]->is_require_grad()) {
                node->require_grad();
                node->edges.push_back(CatEdge0::create(nodes[i], i*shape.colCnt));
            }
        }
        return node;
    }

    Node *cat1(const std::vector<Node *> &nodes) {
        Shape shape = nodes[0]->get_weight()->getShape();
        uint rowCntSum = 0;
        for (uint i = 0; i < nodes.size(); ++ i) {
            // nodes[i]->checkShape(shape);
            assert(nodes[i]->getShape().colCnt == shape.colCnt);
            rowCntSum += nodes[i]->getShape().rowCnt;
        }
        Matrix *m = allocTmpMatrix(Shape(rowCntSum, shape.colCnt));
        Node *node = allocNode(m);
        auto m_buffer = m->getData();
        uint offset = 0;
        for (uint i = 0; i < nodes.size(); ++ i) {
            auto node_i_buffer = nodes[i]->get_weight()->getData();
            memcpy(m_buffer + offset, node_i_buffer, nodes[i]->getShape().size() * sizeof(DATATYPE));
            if (nodes[i]->is_require_grad()) {
                node->require_grad();
                node->edges.push_back(CatEdge1::create(nodes[i], offset));
            }
            offset += nodes[i]->getShape().size();
        }
        return node;
    }

    Node *cat(const std::vector<Node *> &nodes, uint dim) {
        assert(dim == 0 || dim == 1);
        assert(nodes.size() > 0);
        if (dim == 0) { // 这里似乎反了，将错就错，dim == 0 时我们拼接行，split也要这样实现
            return cat0(nodes);
        } else if (dim == 1) {
            return cat1(nodes);
        }
        return nullptr;
    }

    std::vector<Node *> Node::split0() { // 注意这个函数只用在反向传播中，不需要edge
        Shape shape = this->get_weight()->getShape();
         uint colCnt = shape.colCnt;
         uint rowCnt = shape.rowCnt;
         std::vector<Node *> res;
         for (uint i = 0; i < colCnt; ++ i) {
             Matrix *m = allocTmpMatrix(Shape(rowCnt, 1));
             Node *n = allocNode(m);
             if (is_require_grad()) {
                 n->require_grad();
             }
             for (uint j = 0; j < rowCnt; ++ j) {
                 (*m)[j][0] = (*this->get_weight())[j][i];
             }
             res.push_back(n);
         }
         return res;
    }

    std::vector<Node *> Node::split1(uint step) {
        Shape shape = this->get_weight()->getShape();
        uint colCnt = shape.colCnt;
        uint rowCnt = shape.rowCnt;
        assert(step > 0 && rowCnt % step == 0);
        std::vector<Node *> res;
        for (uint i = 0; i < rowCnt; i += step) {
            Matrix *m = allocTmpMatrix(Shape(step, colCnt));
            Node *n = allocNode(m);
            if (is_require_grad()) {
                n->require_grad();
            }
            for (uint j = 0; j < step; ++ j) {
                for (uint k = 0; k < colCnt; ++ k) {
                    (*m)[j][k] = (*this->get_weight())[i+j][k];
                }
            }
            assert(i % step == 0);
            n->edges.push_back(SplitEdge1::create(this, i/step, step));
            res.push_back(n);
        }
        return res;
    }

    std::vector<Node *> Node::split(uint dim, uint step) {
        assert(dim == 0 || dim == 1);
        if (dim == 0) { // 将错就错，dim == 0 时我们切割行，cat也要这样实现
            // assert(false);
            assert(step == 1);
            return split0();
            
        } else if (dim == 1) {
            return split1(step);
        }
        return {};
    }

    void Node::backward() {
        assert(ref_cnt == 0);
        if (!is_require_grad()) {
            return;
        }
        assert(grad != nullptr);
        for (auto edge : edges) {
            edge->node->dec_ref();
            if (!edge->node->is_require_grad()) {
                continue;
            }
            edge->backward(grad);
        }
        for (auto edge : edges) {
            if (edge->node->is_require_grad() && 
                edge->node->get_ref() == 0) {
                edge->node->backward();
            }
        }
    }

    std::vector<Edge *> edges;
    std::vector<Node *> nodes;

    Node *allocNode(Matrix *w) {
        Node *node = new Node(w, true);
        nodes.push_back(node);
        return node;
    }

    void freeAllNodes() {
        for (auto node : nodes) {
            delete node;
        }
        nodes.clear();
    }

    void freeAllEdges() {
        for (auto edge : edges) {
            delete edge;
        }
        edges.clear();
    }

    TmpNodesStats tmpNodesStats() {
        TmpNodesStats stats;
        uint size = nodes.size();
        uint bytes = size * sizeof(Node);
        stats.size = size;
        stats.bytes = bytes;
        return stats;
    }

    TmpEdgesStats tmpEdgesStats() {
        TmpEdgesStats stats;
        uint size = edges.size();
        uint bytes = size * sizeof(Edge);
        stats.size = size;
        stats.bytes = bytes;
        return stats;
    }
} // namespace autograd