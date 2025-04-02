#ifndef AUTOGARD_NODE_H
#define AUTOGARD_NODE_H

#include <assert.h>
#include <vector>

#include "matrix/matrix.h"
#include "stats/stats.h"

namespace autograd {
    class Edge;
    class Node;
    extern std::vector<Edge *> edges;
    extern std::vector<Node *> nodes;

    enum OpType {
        Add,
        Minus,
        ExpandAdd,
        Mul,
        Sub,
        Div,
        MulSV, // single value
        MatMulL,
        MatMulR,
        Tanh,
        Cat0,
        Cat1,
        Split0,
        Split1,
        Sigmoid,
        Relu,
        CrossEntropy,
        CrossEntropyMask,
        Norm,
        Softmax,
        Transpose,
    };
    class Node {
        public:
            Node(Matrix *_w, bool magic = false)
                : w(_w), grad(nullptr), requires_grad(false),
                ref_cnt(0) {
                assert(magic);
            }
            void require_grad() {
                if (!requires_grad) {
                    grad = allocTmpMatrix(w->getShape());
                }
                requires_grad = true;
            }
            bool is_require_grad() const {
                return requires_grad;
            }
            void inc_ref() {
                ref_cnt++;
            }
            void dec_ref() {
                ref_cnt--;
            }
            int get_ref() const {
                return ref_cnt;
            }

            void backward();
            Matrix *get_weight() {
                return w;
            }
            Matrix *get_grad() {
                return grad;
            }

            Shape getShape() {
                return w->getShape();
            }

            void zero_grad() {
                if (requires_grad) {
                    grad = allocTmpMatrix(w->getShape());                
                }
            }

            bool checkShape(const Shape &shape) {
                return w->checkShape(shape);
            }

            Node *operator+(Node *rhs);
            Node *operator*(Node *rhs);
            Node *expand_add(Node *rhs);
            Node *at(Node *rhs);
            Node *Relu();
            Node *CrossEntropy(const std::vector<uint> &labels);
            Node *CrossEntropyMask(const std::vector<uint> &labels, const std::vector<bool> &mask);
            Node *Tanh();
            Node *Sigmoid();
            Node *Norm();
            Node *Softmax();
            Node *Transpose();
            Node *Mul(DATATYPE v);
            Node *Div(DATATYPE v);
            std::vector<Node *> split(uint dim, uint step = 1);
            std::vector<Node *> split0();
            std::vector<Node *> split1(uint step);
            // Node *operator*(Node &rhs);
            // Node *operator/(Node &rhs);
            // Node *operator-(Node &rhs);
            // Node *operator-();
            friend Node *operator-(DATATYPE, Node &rhs);
            friend Node *cat(const std::vector<Node *> &nodes, uint);
            friend Node *cat0(const std::vector<Node *> &nodes);
            friend Node *cat1(const std::vector<Node *> &nodes);
        private:
            Matrix *w;
            Matrix *grad;
            bool requires_grad;
            std::vector<Edge *> edges;
            int ref_cnt;
    };

    Node *cat(const std::vector<Node *> &nodes, uint dim = 0);
    Node *cat0(const std::vector<Node *> &nodes);
    Node *cat1(const std::vector<Node *> &nodes);
    
    Node *allocNode(Matrix *w);
    void freeAllNodes();
    void freeAllEdges();
    TmpNodesStats tmpNodesStats();
    TmpEdgesStats tmpEdgesStats();

    class Edge {
        public:
            Edge(
                OpType _type,
                Node *_node) 
                : type(_type),
                node(_node) {
                    node->inc_ref();
                }
            virtual ~Edge() {};
            virtual void backward(Matrix *grad) = 0;
        protected:
            OpType type;
            Node *node; // node that this edge points to
        friend class Node;
    };

    class AddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new AddEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            AddEdge(Node *_node)
                : Edge(Add, _node) {}
            virtual ~AddEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *grad;
            }
    };

    class MinusEdge: public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new MinusEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            MinusEdge(Node *_node)
                : Edge(Minus, _node) {}
            virtual ~MinusEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() -= *grad;
            }
    };

    class MulEdge : public Edge {
        public:
            static Edge* create(Node *_node, Matrix *_param) {
                Edge *edge = new MulEdge(_node, _param);
                edges.push_back(edge);
                return edge;
            }
            MulEdge(Node *_node, Matrix *_param)
                : Edge(Mul, _node), param(_param) {}
            virtual ~MulEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(param));
            }
        private:
            Matrix *param;
    };

    class ExpandAddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ExpandAddEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            ExpandAddEdge(Node *_node)
                : Edge(ExpandAdd, _node) {}
            virtual ~ExpandAddEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(grad->sum(1));
            }
    };

    class MatMulLEdge : public Edge {
        public:
            static Edge* create(Node *_node, Matrix *_param) {
                Edge *edge = new MatMulLEdge(_node, _param);
                edges.push_back(edge);
                return edge;
            }
            MatMulLEdge(Node *_node, Matrix *_param)
                : Edge(MatMulL, _node), param(_param) {}
            virtual ~MatMulLEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                // *node->get_grad() is grad of W
                *node->get_grad() += *(grad->at(*(param->transpose())));
            }
        private:
            Matrix *param; // Input Vector
    };

    class MatMulREdge : public Edge {
        public:
            static Edge* create(Node *_node, Matrix *_param) {
                Edge *edge = new MatMulREdge(_node, _param);
                edges.push_back(edge);
                return edge;
            }
            MatMulREdge(Node *_node, Matrix *_param)
                : Edge(MatMulR, _node), param(_param) {}
            virtual ~MatMulREdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                // *node->get_grad() is grad of Input
                *node->get_grad() += *(param->transpose()->at(*grad));
            }
        private:
            Matrix *param; // W
    };

    class ReluEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ReluEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            ReluEdge(Node *_node)
                : Edge(Relu, _node) {}
            virtual ~ReluEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(node->get_weight()->Relu_prime()));
            }
    };

    struct CrosEntropyInfo {
        DATATYPE sum, max;
    };

    class CrossEntropyEdge : public Edge {
        public:
            static Edge* create(Node *_node, const std::vector<uint> &_labels, const std::vector<CrosEntropyInfo> &_info) { 
                Edge *edge = new CrossEntropyEdge(_node, _labels, _info);
                edges.push_back(edge);
                return edge;
            }
            CrossEntropyEdge(Node *_node, const std::vector<uint> &_labels, const std::vector<CrosEntropyInfo> &_info)
                : Edge(CrossEntropy, _node), labels(_labels), info(_info) {}
            virtual ~CrossEntropyEdge() {}
            void backward(Matrix *) override {
                assert(node->is_require_grad());
                #pragma omp parallel for
                for (uint i = 0; i < labels.size(); ++i) {
                    auto target = labels[i];
                    DATATYPE max = info[i].max;
                    DATATYPE sum = info[i].sum;
                    for (uint j = 0; j < node->get_weight()->getShape().rowCnt; ++j) {
                        if (j == target) {
                            continue;
                        }
                        auto &_grad = (*node->get_grad())[j][i];
                        _grad = std::exp((*node->get_weight())[j][i] - max) / sum / labels.size();
                    }
                    (*node->get_grad())[target][i] = (std::exp((*node->get_weight())[target][i] - max) / sum - 1) / labels.size();
                }
            }
        private:
            std::vector<uint> labels;
            std::vector<CrosEntropyInfo> info;
    };

    class CrossEntropyMaskEdge: public Edge {
        public:
            static Edge* create(
                Node *_node,
                const std::vector<uint> &_labels,
                const std::vector<bool> &_mask,
                const std::vector<CrosEntropyInfo> &_info) {
                Edge *edge = new CrossEntropyMaskEdge(_node, _labels, _mask, _info);
                edges.push_back(edge);
                return edge;
            }
            CrossEntropyMaskEdge(
                Node *_node,
                const std::vector<uint> &_labels,
                const std::vector<bool> &_mask,
                const std::vector<CrosEntropyInfo> &_info
            ): Edge(CrossEntropyMask, _node), labels(_labels), mask(_mask), info(_info) {}
            virtual ~CrossEntropyMaskEdge() {}
            void backward(Matrix *) override {
                assert(node->is_require_grad());
                uint mask_cnt = 0;
                #pragma omp parallel for reduction(+:mask_cnt)
                for (uint i = 0; i < mask.size(); ++ i) {
                    mask_cnt += mask[i];
                }
                if (mask_cnt == 0) {
                    return;
                }
                #pragma omp parallel for
                for (uint j = 0; j < node->get_weight()->getShape().colCnt; ++ j) {
                    if (!mask[j]) {
                        continue;
                    }
                    auto target = labels[j];
                    DATATYPE max = info[j].max;
                    DATATYPE sum = info[j].sum;
                    for (uint i = 0; i < node->get_weight()->getShape().rowCnt; ++ i) {
                        if (i == target) {
                            continue;
                        }
                        auto &_grad = (*node->get_grad())[i][j];
                        _grad = std::exp((*node->get_weight())[i][j] - max) / sum / mask_cnt;
                    }
                    (*node->get_grad())[target][j] = (std::exp((*node->get_weight())[target][j] - max) / sum - 1) / mask_cnt;
                }
            }
        private:
            std::vector<uint> labels;
            std::vector<bool> mask;
            std::vector<CrosEntropyInfo> info;
    };

    class TanhEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new TanhEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            TanhEdge(Node *_node)
                : Edge(Tanh, _node) {}
            virtual ~TanhEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(node->get_weight()->tanh_prime()));
            }
    };

    class SigmoidEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new SigmoidEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            SigmoidEdge(Node *_node)
                : Edge(Sigmoid, _node) {}
            virtual ~SigmoidEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * *(node->get_weight()->sigmoid_prime()));
            }
    };

    class CatEdge0: public Edge {
        public:
            static Edge* create(Node *_node, uint _offset) {
                Edge *edge = new CatEdge0(_node, _offset);
                edges.push_back(edge);
                return edge;
            }
            CatEdge0(Node *_node, uint _offset)
                : Edge(OpType::Cat0, _node), offset(_offset){}
            virtual ~CatEdge0() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                Shape shape = node->get_weight()->getShape();
                for (uint i = 0; i < shape.rowCnt; ++ i) {
                    for (uint j = 0; j < shape.colCnt; ++ j) {
                        (*node->get_grad())[i][j] += (*grad)[i][j+offset];
                    }
                }
            }
        private:
            uint offset;
    };

    class SplitEdge1: public Edge {
        public:
            static Edge* create(Node *_node, uint _index, uint _step) {
                Edge *edge = new SplitEdge1(_node, _index, _step);
                edges.push_back(edge);
                return edge;
            }
            SplitEdge1(Node *_node, uint _index, uint _step)
                : Edge(OpType::Split1, _node), index(_index), step(_step){}
            virtual ~SplitEdge1() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                assert(node->get_grad()->getShape().colCnt == grad->getShape().colCnt); 
                Shape shape = grad->getShape();
                assert(shape.rowCnt == step);
                uint rowBase = index * step;
                for (uint i = 0; i < shape.rowCnt; i ++) {
                    for (uint j = 0; j < shape.colCnt; ++ j) {
                        assert(i + rowBase < node->get_grad()->getShape().rowCnt);
                        assert(i < shape.rowCnt);
                        assert(j < shape.colCnt);
                        assert(j < node->get_grad()->getShape().colCnt);
                        (*node->get_grad())[i + rowBase][j] += (*grad)[i][j];
                    }
                }
            }
        private:
            uint index;
            uint step;
    };

    class CatEdge1: public Edge {
        public:
            static Edge* create(Node *_node, uint _offset) {
                Edge *edge = new CatEdge1(_node, _offset);
                edges.push_back(edge);
                return edge;
            }
            CatEdge1(Node *_node, uint _offset)
                : Edge(OpType::Cat1, _node), offset(_offset){}
            virtual ~CatEdge1() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                assert(grad->getShape().colCnt == node->getShape().colCnt);
                Shape shape = node->getShape();
                DATATYPE *m_buffer = node->get_grad()->getData();
                DATATYPE *grad_buffer = grad->getData() + offset;
                for (uint i = 0; i < shape.size(); ++ i) {
                    m_buffer[i] += grad_buffer[i];
                }
            }
        private:
            uint offset;
    };

    class NormEdge: public Edge {
        public:
            static Edge* create(
                Node *_node,
                Matrix *w_hat,
                const std::vector<DATATYPE> &_avg_res, const std::vector<DATATYPE> &_var_res,
                DATATYPE eps) {
                Edge *edge = new NormEdge(_node, w_hat, _avg_res, _var_res, eps);
                edges.push_back(edge);
                return edge;
            }
            NormEdge(
                Node *_node,
                Matrix *_w_hat,
                const std::vector<DATATYPE> &_avg_res,
                const std::vector<DATATYPE> &_var_res,
                DATATYPE _eps
            ) : Edge(OpType::Norm, _node),
                w_hat(_w_hat),
                avg_res(_avg_res),
                var_res(_var_res),
                eps(_eps) {}
            virtual ~NormEdge() {}
            void backward(Matrix *grad) override {
                assert(grad->getShape().colCnt == node->getShape().colCnt);
                assert(node->is_require_grad());
                std::vector<Node *> v_w;
                for (uint k = 0; k < grad->getShape().colCnt; k++) {
                    uint rowCnt = grad->getShape().rowCnt;
                    Matrix *mw = allocTmpMatrix(Shape(rowCnt, rowCnt));
                    for (uint i = 0; i < rowCnt; i++) {
                        for (uint j = 0; j < rowCnt; j++) {
                            int eq = i == j;
                            auto sigma = std::sqrt(var_res[k] + eps);
                            auto x_hat_i = (*w_hat)[i][k];
                            auto x_hat_j = (*w_hat)[j][k];
                            // (*mw)[i][j] = (eq - 1.0 / rowCnt - 1.0 / rowCnt * x_hat_i * x_hat_j) / sigma;
                            // 上面的计算精度降低很多，下面的计算精度更高
                            auto part1 = eq * (int)rowCnt - 1 - x_hat_i * x_hat_j;
                            auto part2 = (int)rowCnt * sigma;
                            (*mw)[i][j] = part1 / part2;
                        }
                    }
                    v_w.push_back(allocNode(mw));
                }
                std::vector<Node *> v_grads = allocNode(grad)->split(0);
                assert(v_w.size() == v_grads.size());
                std::vector<Node *> v_res;
                for (uint i = 0; i < v_w.size(); i++) {
                    v_res.push_back(v_w[i]->at(v_grads[i]));
                }
                *node->get_grad() += *(cat(v_res, 0)->get_weight());
            }
        private:
            Matrix *w_hat;
            std::vector<DATATYPE> avg_res;
            std::vector<DATATYPE> var_res;
            DATATYPE eps;
    };

    class SoftmaxEdge: public Edge {
        public:
            static Edge* create(Node *_node, Node *_res) {
                Edge *edge = new SoftmaxEdge(_node, _res);
                edges.push_back(edge);
                return edge;
            }
            SoftmaxEdge(Node *_node, Node *_res)
                : Edge(Softmax, _node), res(_res) {}
            virtual ~SoftmaxEdge() {}
            void backward(Matrix *grad) override {
                assert(grad->getShape().colCnt == node->getShape().colCnt);
                assert(node->is_require_grad());
                Matrix *softmax_grad = allocTmpMatrix(grad->getShape());
                uint rowCnt = grad->getShape().rowCnt;
                uint colCnt = grad->getShape().colCnt;
                for (uint k = 0; k < colCnt; k++) {
                    for (uint target = 0; target < rowCnt; target++) {
                        for (uint i = 0; i < rowCnt; i++) {
                            if (i != target) {
                                (*softmax_grad)[i][k] += -(*res->get_weight())[target][k] * (*res->get_weight())[i][k] * (*grad)[target][k];
                            } else {
                                (*softmax_grad)[i][k] += (*res->get_weight())[i][k] * (1 - (*res->get_weight())[i][k]) * (*grad)[i][k];
                            }
                        }
                    }
                }
                assert(softmax_grad->checkShape(grad->getShape()));
                *node->get_grad() += *softmax_grad;
            }
        private:
            Node *res;
    };

    class TransposeEdge: public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new TransposeEdge(_node);
                edges.push_back(edge);
                return edge;
            }
            TransposeEdge(Node *_node)
                : Edge(OpType::Transpose, _node) {}
            virtual ~TransposeEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(grad->transpose());
            }
    };

    class MulSingleValueEdge: public Edge {
        public:
            static Edge* create(Node *_node, DATATYPE _v) {
                Edge *edge = new MulSingleValueEdge(_node, _v);
                edges.push_back(edge);
                return edge;
            }
            MulSingleValueEdge(Node *_node, DATATYPE _v)
                : Edge(OpType::MulSV, _node), v(_v) {}
            virtual ~MulSingleValueEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad * v);
            }
        private:
            DATATYPE v;
    };

    class DivEdge: public Edge {
        public:
            static Edge* create(Node *_node, DATATYPE _v) {
                Edge *edge = new DivEdge(_node, _v);
                edges.push_back(edge);
                return edge;
            }
            DivEdge(Node *_node, DATATYPE _v)
                : Edge(OpType::Div, _node), v(_v) {}
            virtual ~DivEdge() {}
            void backward(Matrix *grad) override {
                assert(node->is_require_grad());
                *node->get_grad() += *(*grad / v);
            }
        private:
            DATATYPE v;
    };
} // namespace autograd

#endif