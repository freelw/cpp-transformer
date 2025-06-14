#include "node.h"
#include "actions.h"
#include "backends/backend_ops.h"

namespace graph {

    void Node::backward() {
        if (backward_times > 0) {
            return;
        }
        assert(ref_cnt == 0);
        backward_times++;
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

    Node* Node::transpose(int a, int b) {
        Tensor* l_tensor = this->get_tensor();
        Tensor* res_tensor = l_tensor->transpose(a, b);
        Node* res_node = nullptr;
        if (is_require_grad()) {
            res_node = allocNode(res_tensor, this->get_grad()->transpose(a, b));
            res_node->require_grad();
            res_node->edges.push_back(EmptyEdge::create(this));
        }
        else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node* Node::permute(const std::vector<int>& dims) {
        Tensor* l_tensor = this->get_tensor();
        Tensor* res_tensor = l_tensor->permute(dims);
        Node* res_node = nullptr;
        if (is_require_grad()) {
            res_node = allocNode(res_tensor, this->get_grad()->permute(dims));
            res_node->require_grad();
            res_node->edges.push_back(EmptyEdge::create(this));
        }
        else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node* Node::reshape(const std::vector<int>& shape) {
        Tensor* l_tensor = this->get_tensor();
        Tensor* res_tensor = l_tensor->reshape(shape);
        bool share_mem = l_tensor->is_shared_with(res_tensor);
        assert(l_tensor->is_contiguous() == share_mem);
        Node* res_node = nullptr;
        if (is_require_grad()) {
            Tensor* grad = this->get_grad();
            Tensor* res_grad = grad->reshape(shape);
            assert(grad->is_contiguous() == share_mem);
            assert(grad->is_shared_with(res_grad) == share_mem);
            res_node = allocNode(res_tensor, res_grad);
            res_node->require_grad();
            if (share_mem) {
                res_node->edges.push_back(EmptyEdge::create(this));
            }
            else {
                res_node->edges.push_back(ReshapeEdge::create(this));
            }
        }
        else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node* Node::sequence_mask(Tensor* mask, float value) {
        Tensor* res_tensor = this->get_tensor()->sequence_mask(mask, value);
        Node* res_node = nullptr;
        if (is_require_grad()) {
            res_node = allocNode(res_tensor, this->get_grad());
            res_node->edges.push_back(EmptyEdge::create(this));
        }
        else {
            res_node = allocNode(res_tensor);
        }
        return res_node;
    }

    Node* Node::softmax() {
        assert(this->get_tensor()->get_dim() == 3);
        auto shape = this->get_tensor()->get_shape();
        auto res_node = allocNode(
            this->get_tensor()->softmax()
        );
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(SoftmaxEdge::create(this, res_node->get_tensor()));
        }
        return res_node;
    }

    Node* Node::masked_softmax(Tensor* valid_len) {
        assert(this->get_tensor()->get_dim() == 3);
        if (valid_len == nullptr) {
            return this->softmax();
        }
        else {
            auto shape = this->get_tensor()->get_shape();
            Tensor* mask = valid_len->get_dim() == 1 ?
                valid_len->repeat_interleave(shape[1]) : valid_len->reshape({ -1 });
            auto reshape1_res = this->reshape({ -1, shape[2] });
            auto reshape1_res_shape = reshape1_res->get_tensor()->get_shape();
            auto sequence_mask_res = reshape1_res->sequence_mask(mask, -1e6f);
            auto reshape2_res = sequence_mask_res->reshape(shape);
            auto softmax_res = reshape2_res->softmax();
            return softmax_res;
        }
    }
    Node* Node::add(Node* rhs) {
        Tensor* res_tensor = callocTensor(t->get_shape(), "add_res");
        Tensor* r_tensor = rhs->get_tensor();
        gCreateAction(
            new AddAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(AddEdge::create(rhs));
            }
        }
        return res_node;
    }

    Node* Node::mul(Node* rhs) {
        Tensor* res_tensor = callocTensor(t->get_shape(), "mul_res");
        Tensor* r_tensor = rhs->get_tensor();
        gCreateAction(
            new MulAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(MulEdge::create(this, rhs));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(MulEdge::create(rhs, this));
            }
        }
        return res_node;
    }

    Node* Node::mulsv(float v) {
        Tensor* l_tensor = this->get_tensor();
        assert(l_tensor->is_contiguous()); // 只有在这个前提下，当前的后端实现才是正确的，没有考虑stride
        Tensor* res_tensor = callocTensor(t->get_shape(), "mulsv_res");
        gCreateAction(
            new MulSVAction(
                l_tensor,
                res_tensor,
                v
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(MulSVEdge::create(this, v));
        }
        return res_node;
    }

    Node* Node::expand_add(Node* rhs) {
        Tensor* res_tensor = callocTensor(
            t->get_shape(),
            this->get_tensor()->get_name() + "_" +
            rhs->get_tensor()->get_name() +
            "_expand_add_res"
        );
        Tensor* r_tensor = rhs->get_tensor();
        assert(r_tensor->get_dim() == 1);
        gCreateAction(
            new ExpandAddAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(AddEdge::create(this));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(ExpandAddEdge::create(rhs));
            }
        }
        return res_node;
    }

    Node* Node::expand_mul(Node* rhs) {
        Tensor* res_tensor = callocTensor(t->get_shape(), "expand_mul");
        Tensor* r_tensor = rhs->get_tensor();
        assert(r_tensor->get_dim() == 1);
        gCreateAction(
            new ExpandMulAction(
                this->get_tensor(),
                rhs->get_tensor(),
                res_tensor
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (is_require_grad()) {
                res_node->edges.push_back(ExpandMulEdgeL::create(this, rhs));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(ExpandMulEdgeR::create(rhs, this));
            }
        }
        return res_node;
    }

    void atImpl(Node* lhs, Node* rhs, Node* res_node) {
        Tensor* l_tensor = lhs->get_tensor();
        Tensor* r_tensor = rhs->get_tensor();
        Tensor* res_tensor = res_node->get_tensor();
        gCreateAction(
            new AtAction(
                l_tensor,
                r_tensor,
                res_tensor
            )
        );
        if (lhs->is_require_grad() || rhs->is_require_grad()) {
            res_node->require_grad();
            if (lhs->is_require_grad()) {
                res_node->edges.push_back(MatMulLEdge::create(lhs, rhs));
            }
            if (rhs->is_require_grad()) {
                res_node->edges.push_back(MatMulREdge::create(rhs, lhs));
            }
        }
    }

    Node* Node::at(Node* rhs) {
        Tensor* r_tensor = rhs->get_tensor();
        Tensor* l_tensor = this->get_tensor();
        assert(l_tensor->get_dim() == 2);
        assert(r_tensor->get_dim() == 2);
        assert(l_tensor->get_shape()[1] == r_tensor->get_shape()[0]);

        Tensor* res_tensor = callocTensor({ l_tensor->get_shape()[0], r_tensor->get_shape()[1] }, "res_at");
        Node* res_node = allocNode(res_tensor);
        atImpl(this, rhs, res_node);
        return res_node;
    }

    Node* Node::bmm(Node* rhs) {
        Node* l_node = this;
        Node* r_node = rhs;

        if (!l_node->get_tensor()->is_contiguous()) {
            l_node = l_node->reshape(l_node->get_tensor()->get_shape());
        }
        if (!r_node->get_tensor()->is_contiguous()) {
            r_node = r_node->reshape(r_node->get_tensor()->get_shape());
        }

        auto l_tensor = l_node->get_tensor();
        auto r_tensor = r_node->get_tensor();
        assert(l_tensor->get_dim() == 3);
        assert(r_tensor->get_dim() == 3);
        assert(l_tensor->get_shape()[2] == r_tensor->get_shape()[1]);

        if (l_node->is_require_grad()) {
            auto l_grad = l_node->get_grad();
            assert(l_tensor->get_shape() == l_grad->get_shape());
            assert(l_grad->get_dim() == 3);
            assert(l_tensor->is_contiguous() == l_grad->is_contiguous());
        }

        if (r_node->is_require_grad()) {
            auto r_grad = r_node->get_grad();
            assert(r_tensor->get_shape() == r_grad->get_shape());
            assert(r_grad->get_dim() == 3);
            assert(r_tensor->is_contiguous() == r_grad->is_contiguous());
        }
        std::vector<Node*> l_split_2d_nodes;
        std::vector<Node*> r_split_2d_nodes;

        l_node->split_3d(l_split_2d_nodes);
        r_node->split_3d(r_split_2d_nodes);

        assert(l_split_2d_nodes.size() == r_split_2d_nodes.size());
        Tensor* res_tensor = callocTensor(
            { l_tensor->get_shape()[0], l_tensor->get_shape()[1], r_tensor->get_shape()[2] },
            l_tensor->get_name() + "_" + r_tensor->get_name() + "_bmm_res"
        );
        Node* res_node = allocNode(res_tensor);
        if (l_node->is_require_grad() || r_node->is_require_grad()) {
            res_node->require_grad();
        }
        std::vector<Node*> res_nodes;
        res_node->split_3d(res_nodes, true); // opposite = true
        assert(res_nodes.size() == l_split_2d_nodes.size());
        for (int i = 0; i < l_split_2d_nodes.size(); ++i) {
            Node* l_split_2d_node = l_split_2d_nodes[i];
            Node* r_split_2d_node = r_split_2d_nodes[i];
            Node* res_split_2d_node = res_nodes[i];
            atImpl(
                l_split_2d_node,
                r_split_2d_node,
                res_split_2d_node
            );
        }
        return res_node;
    }

    void Node::split_3d(std::vector<Node*>& res_nodes, bool opposite) {
        assert(this->get_tensor()->get_dim() == 3);
        if (this->is_require_grad()) {
            assert(this->get_grad()->get_dim() == 3);
        }
        auto shape = this->get_tensor()->get_shape();
        res_nodes.clear();
        res_nodes.reserve(shape[0]);
        int offset = 0;
        int block = shape[1] * shape[2];
        for (int i = 0; i < shape[0]; ++i) {
            Node* node = nullptr;
            std::vector<int> new_strides;
            new_strides.resize(2);
            new_strides[0] = shape[2];
            new_strides[1] = 1;
            Tensor* new_tensor = allocTensorView(
                this->get_tensor(),
                { shape[1], shape[2] },
                new_strides,
                this->get_tensor()->get_name() + "_split_" + std::to_string(i),
                offset
            );
            if (this->is_require_grad()) {
                Tensor* new_grad = allocTensorView(
                    this->get_grad(),
                    { shape[1], shape[2] },
                    new_strides,
                    this->get_grad()->get_name() + "_split_" + std::to_string(i),
                    offset
                );
                node = allocNode(new_tensor, new_grad);
                if (opposite) { // 考虑split的左操作数是结果，梯度需要从整个结果传递给子结果
                    this->edges.push_back(EmptyEdge::create(node));
                }
                else { // 考虑split的左操作数是输入，梯度需要从子结果传递给整个结果
                    node->edges.push_back(EmptyEdge::create(this));
                }
            }
            else {
                node = allocNode(new_tensor);
            }
            offset += block;
            res_nodes.push_back(node);
        }
    }

    Node* Node::relu() {
        Tensor* l_tensor = this->get_tensor();
        Tensor* res_tensor = callocTensor(l_tensor->get_shape(), "relu_res");
        gCreateAction(
            new ReluAction(
                l_tensor,
                res_tensor
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(ReluEdge::create(this));
        }
        return res_node;
    }

    Node* Node::norm() {
        assert(this->get_tensor()->get_dim() == 2);
        auto shape = this->get_tensor()->get_shape();

        Tensor* avg_tensor = callocTensor({ shape[0] }, "avg");
        Tensor* var_tensor = callocTensor({ shape[0] }, "var");

        gCreateAction(
            new AvgAction(
                this->get_tensor(),
                avg_tensor
            )
        );

        gCreateAction(
            new VarAction(
                this->get_tensor(),
                avg_tensor,
                var_tensor
            )
        );

        Tensor* norm_res = callocTensor(
            this->get_tensor()->get_shape(),
            "norm_res"
        );

        gCreateAction(
            new NormAction(
                this->get_tensor(),
                avg_tensor,
                var_tensor,
                norm_res
            )
        );

        auto res_node = allocNode(norm_res);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(NormEdge::create(this, norm_res, var_tensor));
        }
        return res_node;
    }

    Node* Node::avg_1d(Tensor* mask) {
        Tensor* l_tensor = this->get_tensor();
        assert(l_tensor->get_dim() == 1);
        if (mask == nullptr) {
            mask = callocTensor({ l_tensor->get_shape()[0] }, "avg_1d_mask");
            gCreateAction(
                new FillWeightAction(
                    mask,
                    "fill",
                    1.0f,
                    0
                )
            );
        }
        assert(mask->get_dim() == 1);
        mask = mask->reshape({ -1, 1 });
        auto shape = l_tensor->get_shape();
        l_tensor = l_tensor->reshape({ 1, shape[0] });
        Tensor* res_tensor = callocTensor({ 1 }, "avg_1d_res");
        Tensor* sum_tensor = callocTensor({ 1 }, "avg_1d_sum");
        Tensor* mask_sum_tensor = callocTensor({ 1 }, "avg_1d_mask_sum");
        gCreateAction(
            new SumAction(
                l_tensor->reshape({ -1, 1 }),
                sum_tensor,
                0
            )
        );
        gCreateAction(
            new SumAction(
                mask,
                mask_sum_tensor,
                0
            )
        );
        gCreateAction(
            new LazyDivAction(
                sum_tensor,
                res_tensor,
                mask_sum_tensor
            )
        );
        auto res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            // std::cout << "this meta : " << this->get_tensor()->get_meta_info() << std::endl;
            res_node->edges.push_back(Avg1dEdge::create(this, mask_sum_tensor));
        }
        return res_node;
    }

    Node* Node::CrossEntropy(Tensor* labels) {
        assert(labels->get_dim() == 1);
        assert(
            labels->get_dtype() == INT32
        );
        assert(labels->get_shape()[0] == this->get_tensor()->get_shape()[0]);
        Tensor* tensor_maxs = callocTensor(labels->get_shape(), "maxs");
        Tensor* tensor_sums = callocTensor(labels->get_shape(), "sums");
        Tensor* ce_res = callocTensor({ labels->get_shape() }, "cross_entropy");

        gCreateAction(
            new CrossEntropyAction(
                this->get_tensor(),
                labels,
                tensor_maxs,
                tensor_sums,
                ce_res
            )
        );
        Node* res_node = allocNode(ce_res);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(CrossEntropyEdge::create(this, labels, tensor_maxs, tensor_sums));
        }
        return res_node;
    }

    Node* Node::div(float value) {
        Tensor* l_tensor = this->get_tensor();
        assert(l_tensor->is_contiguous()); // 只有在这个前提下，当前的后端实现才是正确的，没有考虑stride
        Tensor* res_tensor = callocTensor(l_tensor->get_shape(), "div_res");
        Tensor* value_tensor = callocTensor(
            { 1 },
            "div_value",
            FLOAT32
        );
        gCreateAction(
            new AssignValueAction(
                value_tensor,
                value
            )
        );
        // g_backend_ops->cp_to_device(
        //     value_tensor,
        //     (char*)&value,
        //     sizeof(float)
        // );
        gCreateAction(
            new LazyDivAction(
                l_tensor,
                res_tensor,
                value_tensor
            )
        );
        Node* res_node = allocNode(res_tensor);
        if (is_require_grad()) {
            res_node->require_grad();
            res_node->edges.push_back(DivEdge::create(this, value_tensor));
        }
        return res_node;
    }

    Node* Node::mask(Tensor* m) {
        Tensor* l_tensor = this->get_tensor();
        assert(l_tensor->get_shape() == m->get_shape());
        auto nm = allocNode(m);
        return this->mul(nm);
    }

    void Node::init_weight_gauss(float sigma, float mean) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "gauss",
                sigma,
                mean
            )
        );
    }

    void Node::init_weight_uniform(float sigma) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "uniform",
                sigma,
                0
            )
        );
    }

    void Node::init_weight_for_dbg(float scale) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "dbg",
                scale,
                0
            )
        );
    }

    void Node::init_weight_fill(float value) {
        gCreateAction(
            new InitWeightAction(
                this->get_tensor(),
                "fill",
                value,
                0
            )
        );
    }

    void CrossEntropyEdge::backward(Tensor* grad) {
        Tensor* tmp = callocTensor(
            node->get_grad()->get_shape(),
            "cross_entropy_tmp"
        );
        gCreateAction(
            new CrossEntropyBackwardAction(
                node->get_tensor(),
                labels,
                maxs,
                sums,
                tmp
            )
        );

        Tensor* tmp2 = callocTensor(
            node->get_grad()->get_shape(),
            "cross_entropy_tmp2"
        );
        gCreateAction(
            new ExpandMulAction(
                tmp->transpose(),
                grad,
                tmp2->transpose()
            )
        );

        gCreateAction(
            new AddEqAction(
                node->get_grad(),
                tmp2
            )
        );
    }

    void SoftmaxEdge::backward(Tensor* grad) {

        Tensor* tmp = callocTensor(
            node->get_grad()->get_shape(),
            "softmax_tmp"
        );

        gCreateAction(
            new SoftmaxBackwardAction(
                tmp,
                softmax_res,
                grad
            )
        );
        gCreateAction(
            new AddEqAction(
                node->get_grad(),
                tmp
            )
        );
    }

    void EmbeddingEdge::backward(Tensor* grad) {

        Tensor* tmp = callocTensor(
            node->get_grad()->get_shape(),
            "embedding_tmp"
        );

        gCreateAction(
            new EmbeddingBackwardAction(
                grad,
                indices,
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

    void NormEdge::backward(Tensor* grad) {
        Tensor* tmp = callocTensor(
            node->get_grad()->get_shape(),
            "norm_tmp"
        );

        gCreateAction(
            new NormBackwardAction(
                grad,
                norm_res,
                var_res,
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

    std::vector<Edge*> edges;
    std::vector<Node*> nodes;
#ifndef NDEBUG
    std::vector<Node*> g_dbg_nodes;
#endif

    Node* allocNode(Tensor* t) {
        Node* node = new Node(t);
        nodes.push_back(node);
        return node;
    }

    Node* allocNode(Tensor* t, Tensor* grad) {
        Node* node = new Node(t, grad);
        nodes.push_back(node);
        return node;
    }

    void validateAllNodes() {
        for (Node* node : nodes) {
            if (node->is_require_grad()) {
                bool grad_contiguous = node->get_grad()->is_contiguous();
                bool tensor_contiguous = node->get_tensor()->is_contiguous();
                assert(grad_contiguous == tensor_contiguous);
                auto grad_shape = node->get_grad()->get_shape();
                auto tensor_shape = node->get_tensor()->get_shape();
                assert(grad_shape == tensor_shape);
            }
        }
    }

    void validateAllNodesRefCnt(int cnt) {
        bool succ = true;
        for (Node* node : nodes) {
            if (node->get_ref() != cnt) {
                std::cerr << "Node " << node->get_tensor()->get_name()
                    << " has non-" << cnt << " ref count: " << node->get_ref() << std::endl;
                succ = false;
            }
        }
        assert(succ);
    }

    void validateAllNodesGradZero() {
        for (Node* node : nodes) {
            if (node->is_require_grad()) {
                char* buffer = static_cast<char*>(::malloc(node->get_grad()->size()));
                g_backend_ops->cp_from_device(
                    buffer,
                    node->get_grad(),
                    node->get_grad()->size()
                );
                for (int i = 0; i < node->get_grad()->size(); ++i) {
                    if (buffer[i] != char(0)) {
                        std::cerr << "tensor " << node->get_grad()->get_name()
                            << " grad is not zero at index " << i
                            << ", value: " << int(buffer[i]) << std::endl;
                        assert(false);
                    }
                }
                ::free(buffer);
            }
        }
    }

    void gAddEdge(Edge* edge) {
        edges.push_back(edge);
    }

    void freeAllNodes() {
        for (Node* node : nodes) {
            delete node;
        }
        nodes.clear();
    }

    void freeAllEdges() {
        for (Edge* edge : edges) {
            delete edge;
        }
        edges.clear();
    }

} // namespace graph