#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "tensor/tensor.h"
#include <vector>
#include "actions.h"

class Dropout;
class Embedding;

namespace graph {
    class Edge;
    void gAddEdge(Edge *edge);
    class Node {
        public:
            Node(Tensor *_t)
                : t(_t),
                ref_cnt(0),
                b_require_grad(false),
                grad(nullptr),
                backward_times(0) {
                
            }
            Node(Tensor *_t, Tensor *_grad)
                : t(_t),
                ref_cnt(0),
                b_require_grad(true),
                grad(_grad),
                backward_times(0) {
                
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
            Tensor *get_tensor() const {
                return t;
            }
            Tensor *get_grad() const {
                return grad;
            }
            bool is_require_grad() const {
                return b_require_grad;
            }
            void require_grad(bool _b_require_grad = true) {
                if (b_require_grad == _b_require_grad) {
                    return;
                }
                if (_b_require_grad) {
                    grad = allocGradTensor(t->get_shape(), t->get_name()+"_grad");
                } else {
                    grad = nullptr;
                }
                b_require_grad = _b_require_grad;
            }
            void backward();
            Node *transpose(int a = 0, int b = 1);
            Node *permute(const std::vector<int> &dims);
            Node *reshape(const std::vector<int> &shape);
            Node *sequence_mask(Tensor *mask, float value);
            Node *softmax();
            Node *masked_softmax(Tensor *valid_len);
            Node *add(Node *rhs);
            Node *mul(Node *rhs);
            Node *mulsv(float v);
            Node *expand_add(Node *rhs);
            Node *expand_mul(Node *rhs);
            Node *at(Node *rhs);
            Node *bmm(Node *rhs);
            void split_3d(std::vector<Node *> &res_nodes, bool opposite = false);
            Node *relu();
            Node *norm();
            Node *avg_1d(Tensor *mask = nullptr);
            Node *CrossEntropy(Tensor *labels);
            Node *div(float value);
            Node *mask(Tensor *m);
            void init_weight_gauss(float sigma, float mean);
            void init_weight_uniform(float sigma);
            void init_weight_for_dbg(float scale = 1.0f);
            void init_weight_fill(float value);
            friend void atImpl(Node *lhs, Node *rhs, Node *res_node);
        private:
            Tensor *t;
            Tensor *grad;
            std::vector<Edge *> edges;
            int ref_cnt;
            bool b_require_grad;
            int backward_times;
        friend class ::Dropout;
        friend class ::Embedding;
    };

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
        Empty,
        Reshape,
        Embedding,
        ExpandMulL,
        ExpandMulR,
        Avg1d,
        Dropout
    };

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
            virtual void backward(Tensor *grad) = 0;
        protected:
            OpType type;
            Node *node; // node that this edge points to
        friend class Node;
    };

    class AddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new AddEdge(_node);
                graph::gAddEdge(edge);
                return edge;
            }
            AddEdge(Node *_node)
                : Edge(Add, _node) {}
            virtual ~AddEdge() {}
            void backward(Tensor *grad) override {
                gCreateAction(
                    new AddEqAction(node->get_grad(), grad)
                );
            }
    };

    class MulEdge : public Edge {
        public:
            static Edge* create(Node *_node, Node *_rhs) {
                Edge *edge = new MulEdge(_node, _rhs);
                graph::gAddEdge(edge);
                return edge;
            }
            MulEdge(Node *_node, Node *_rhs)
                : Edge(Mul, _node), rhs(_rhs) {}
            virtual ~MulEdge() {}
            void backward(Tensor *grad) override {
                Tensor *tmp = callocTensor(
                    node->get_grad()->get_shape(),
                    "mul_tmp"
                );
                gCreateAction(
                    new MulAction(
                        grad,
                        rhs->get_tensor(),
                        tmp
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            Node *rhs;
    };

    class ExpandAddEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ExpandAddEdge(_node);
                graph::gAddEdge(edge);
                return edge;
            }
            ExpandAddEdge(Node *_node)
                : Edge(ExpandAdd, _node) {}
            virtual ~ExpandAddEdge() {}
            void backward(Tensor *grad) override {
                assert(grad->get_shape().size() == 2);
                std::vector<int> shape = {grad->get_shape()[1]};
                Tensor *tmp = allocGradTensor(shape, "sum_tmp"); // 行向量
                
                gCreateAction(
                    new SumAction(
                        grad,
                        tmp,
                        0
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
    };

    class ExpandMulEdgeL : public Edge {
        public:
            static Edge* create(Node *_node, Node *_rhs) {
                assert(_rhs->get_tensor()->get_dim() == 1);
                assert(_node->get_tensor()->get_dim() == 2);
                Edge *edge = new ExpandMulEdgeL(_node, _rhs);
                graph::gAddEdge(edge);
                return edge;
            }
            ExpandMulEdgeL(Node *_node, Node *_rhs)
                : Edge(ExpandMulL, _node), rhs(_rhs) {}
            virtual ~ExpandMulEdgeL() {}
            void backward(Tensor *grad) override {
                Tensor *tmp = callocTensor(
                    grad->get_shape(),
                    "expand_mul_l_add_tmp"
                );
                gCreateAction(
                    new ExpandMulAction(grad, rhs->get_tensor(), tmp)
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            Node *rhs;
    };

    class ExpandMulEdgeR : public Edge {
        public:
            static Edge* create(Node *_node, Node *_rhs) {
                assert(_node->get_tensor()->get_dim() == 1);
                assert(_rhs->get_tensor()->get_dim() == 2);
                Edge *edge = new ExpandMulEdgeR(_node, _rhs);
                graph::gAddEdge(edge);
                return edge;
            }
            ExpandMulEdgeR(Node *_node, Node *_rhs)
                : Edge(ExpandMulR, _node), rhs(_rhs) {}
            virtual ~ExpandMulEdgeR() {}
            void backward(Tensor *grad) override {
                Tensor *tmp = callocTensor(
                    grad->get_shape(),
                    "expand_mul_r_tmp"
                );
                gCreateAction(
                    new MulAction(
                        rhs->get_tensor(),
                        grad,
                        tmp
                    )
                );
                Tensor *sum_tmp = callocTensor(
                    node->get_tensor()->get_shape(),
                    "expand_mul_r_sum_tmp"
                );
                gCreateAction(
                    new SumAction(
                        tmp,
                        sum_tmp,
                        0
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        sum_tmp
                    )
                );
            }
        private:
            Node *rhs;
    };

    class MatMulLEdge : public Edge {
        public:
            static Edge* create(Node *_node, Node *_rhs) {
                Edge *edge = new MatMulLEdge(_node, _rhs);
                gAddEdge(edge);
                return edge;
            }
            MatMulLEdge(Node *_node, Node *_rhs)
                : Edge(MatMulL, _node), rhs(_rhs) {}
            virtual ~MatMulLEdge() {}
            void backward(Tensor *grad) override {
                Tensor *r_tensor = rhs->get_tensor();
                Tensor *l_tensor = node->get_tensor();
                Tensor *r_transpose_view = r_tensor->transpose();
                Tensor *tmp = callocTensor(
                    node->get_grad()->get_shape(),
                    "matmul_l_tmp"
                );
                gCreateAction(
                    new AtAction(
                        grad,
                        r_transpose_view,
                        tmp
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            Node *rhs;
    };

    class MatMulREdge : public Edge {
        public:
            static Edge* create(Node *_node, Node *_lhs) {
                Edge *edge = new MatMulREdge(_node, _lhs);
                gAddEdge(edge);
                return edge;
            }
            MatMulREdge(Node *_node, Node *_lhs)
                : Edge(MatMulR, _node), lhs(_lhs) {}
            virtual ~MatMulREdge() {}
            void backward(Tensor *grad) override {
                Tensor *l_tensor = lhs->get_tensor();
                Tensor *r_tensor = node->get_tensor();
                Tensor *l_transpose_view = l_tensor->transpose();
                
                Tensor *tmp = callocTensor(
                    node->get_grad()->get_shape(),
                    "matmul_r_tmp"
                );
                gCreateAction(
                    new AtAction(
                        l_transpose_view,
                        grad,
                        tmp
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            Node *lhs;
    };

    class ReluEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ReluEdge(_node);
                gAddEdge(edge);
                return edge;
            }
            ReluEdge(Node *_node)
                : Edge(Relu, _node) {}
            virtual ~ReluEdge() {}
            void backward(Tensor *grad) override {
                Tensor *relu_prime_tensor = callocTensor(
                    node->get_tensor()->get_shape(),
                    "relu_prime"
                );

                gCreateAction(
                    new ReluPrimeAction(
                        node->get_tensor(),
                        relu_prime_tensor
                    )
                );

                Tensor *grad_mul_relu_prime = callocTensor(
                    node->get_tensor()->get_shape(),
                    "grad_mul_relu_prime"
                );

                gCreateAction(
                    new MulAction(
                        relu_prime_tensor,
                        grad,
                        grad_mul_relu_prime
                    )
                );

                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        grad_mul_relu_prime
                    )
                );
            }
    };

    class CrossEntropyEdge : public Edge {
        public:
            static Edge* create(
                Node *_node, Tensor *_labels,
                Tensor *_maxs, Tensor *_sums
            ) { 
                Edge *edge = new CrossEntropyEdge(_node, _labels, _maxs, _sums);
                gAddEdge(edge);
                return edge;
            }
            CrossEntropyEdge(
                Node *_node, Tensor *_labels,
                Tensor *_maxs, Tensor *_sums
            ) : Edge(CrossEntropy, _node), labels(_labels), maxs(_maxs), sums(_sums){}
            virtual ~CrossEntropyEdge() {}
            void backward(Tensor *) override;
        private:
            Tensor *labels;
            Tensor *maxs;
            Tensor *sums;
    };

    class EmptyEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new EmptyEdge(_node);
                gAddEdge(edge);
                return edge;
            }
            EmptyEdge(Node *_node)
                : Edge(Empty, _node) {}
            virtual ~EmptyEdge() {}
            void backward(Tensor *grad) override {
            }        
    };

    class ReshapeEdge : public Edge {
        public:
            static Edge* create(Node *_node) {
                Edge *edge = new ReshapeEdge(_node);
                gAddEdge(edge);
                return edge;
            }
            ReshapeEdge(Node *_node)
                : Edge(Reshape, _node) {}
            virtual ~ReshapeEdge() {}
            void backward(Tensor *grad) override {
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        grad->reshape(node->get_grad()->get_shape())
                    )
                );
            }        
    };

    class SoftmaxEdge : public Edge {
        public:
            static Edge* create(Node *_node, Tensor *_softmax_res) {
                Edge *edge = new SoftmaxEdge(_node, _softmax_res);
                gAddEdge(edge);
                return edge;
            }
            SoftmaxEdge(Node *_node, Tensor *_softmax_res)
                : Edge(Softmax, _node), softmax_res(_softmax_res) {}
            virtual ~SoftmaxEdge() {}
            void backward(Tensor *grad) override;
        private:
            Tensor *softmax_res;
    };

    class DivEdge : public Edge {
        public:
            static Edge* create(Node *_node, float value) {
                Edge *edge = new DivEdge(_node, value);
                gAddEdge(edge);
                return edge;
            }
            DivEdge(Node *_node, float value)
                : Edge(Div, _node), value(value) {}
            virtual ~DivEdge() {}
            void backward(Tensor *grad) override {
                Tensor *tmp = callocTensor(
                    node->get_tensor()->get_shape(),
                    "div_tmp"
                );
                gCreateAction(
                    new DivAction(
                        grad,
                        tmp,
                        value
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            float value;
    };

    class DropoutEdge : public Edge {
        public:
            static Edge* create(Node *_node, Tensor *_mask) {
                Edge *edge = new DropoutEdge(_node, _mask);
                gAddEdge(edge);
                return edge;
            }
            DropoutEdge(Node *_node, Tensor *_mask)
                : Edge(Dropout, _node), mask(_mask) {}
            virtual ~DropoutEdge() {}
            void backward(Tensor *grad) override {
                Tensor *tmp = callocTensor(
                    node->get_grad()->get_shape(),
                    "dropout_tmp"
                );
                gCreateAction(
                    new MulAction(
                        grad,
                        mask,
                        tmp
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            Tensor *mask;
    };

    class EmbeddingEdge: public Edge {
        public:
            static Edge* create(Node *_node, Tensor *_indices) {
                Edge *edge = new EmbeddingEdge(_node, _indices);
                gAddEdge(edge);
                return edge;
            }
            EmbeddingEdge(Node *_node, Tensor *_indices)
                : Edge(Embedding, _node), indices(_indices) {}
            virtual ~EmbeddingEdge() {}
            void backward(Tensor *grad) override;
        private:
            Tensor *indices;
    };

    class NormEdge: public Edge {
        public:
            static Edge* create(
                Node *_node, Tensor *_norm_res, Tensor *_var_res) {
                Edge *edge = new NormEdge(_node, _norm_res, _var_res);
                gAddEdge(edge);
                return edge;
            }
            NormEdge(
                Node *_node,
                Tensor *_norm_res,
                Tensor *_var_res
            ) : Edge(Norm, _node), norm_res(_norm_res), var_res(_var_res) {}
            virtual ~NormEdge() {}
            void backward(Tensor *grad) override;
        private:
            Tensor *norm_res;
            Tensor *var_res;
    };

    class Avg1dEdge: public Edge {
        public:
            static Edge* create(Node *_node, Tensor *_mask_sum_tensor) {
                Edge *edge = new Avg1dEdge(_node, _mask_sum_tensor);
                gAddEdge(edge);
                return edge;
            }
            Avg1dEdge(Node *_node, Tensor *_mask_sum_tensor)
                : Edge(Avg1d, _node), mask_sum_tensor(_mask_sum_tensor) {}
            virtual ~Avg1dEdge() {}
            void backward(Tensor *grad) override {
                assert(grad->get_dim() == 1);
                auto shape = grad->get_shape();
                assert(shape[0] == 1);
                Tensor *tmp = callocTensor(
                    node->get_grad()->get_shape(),
                    "avg_1d_tmp_grad"
                );
                
                gCreateAction(
                    new FillWeightAction(
                        tmp,
                        "fill",
                        1.0f,
                        0
                    )
                );
                gCreateAction(
                    new LazyDivAction(
                        tmp,
                        tmp,
                        mask_sum_tensor
                    )
                );
                
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            Tensor *mask_sum_tensor;
    };

    class MulSVEdge: public Edge {
        public:
            static Edge* create(Node *_node, float value) {
                Edge *edge = new MulSVEdge(_node, value);
                gAddEdge(edge);
                return edge;
            }
            MulSVEdge(Node *_node, float value)
                : Edge(MulSV, _node), value(value) {}
            virtual ~MulSVEdge() {}
            void backward(Tensor *grad) override {
                Tensor *tmp = callocTensor(
                    node->get_grad()->get_shape(),
                    "mul_sv_tmp"
                );
                gCreateAction(
                    new MulSVAction(
                        grad,
                        tmp,
                        value
                    )
                );
                gCreateAction(
                    new AddEqAction(
                        node->get_grad(),
                        tmp
                    )
                );
            }
        private:
            float value;
    };

    extern std::vector<Node *> g_dbg_nodes;

    Node *allocNode(Tensor *t);
    Node *allocNode(Tensor *t, Tensor *grad);
    void validateAllNodes();
    void validateAllNodesGradZero();
    void validateAllNodesRefCnt(int cnt = 0);
    void freeAllNodes();
    void freeAllEdges();
}
#endif