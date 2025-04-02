
#include "test.h"
#include "layernorm.h"
#include "attention.h"
#include "posencoding.h"
#include "addnorm.h"
#include "ffn.h"
#include "encoder.h"
#include "decoder.h"

void test_layernorm() {
    auto normalized_shape = 6;
    LayerNorm layernorm(normalized_shape);
    Matrix *input = allocTmpMatrix(Shape(normalized_shape, 2));
    for (int i = 0; i < normalized_shape; i++) {
        (*input)[i][0] = i;
        (*input)[i][1] = i+1;
    }
    std::vector<uint> labels = {2, 3};
    autograd::Node *x = autograd::allocNode(input);
    x->require_grad();
    autograd::Node *y = layernorm.forward(x);

    auto loss = y->CrossEntropy(labels);
    cout << "y: " << endl;
    cout << *y->get_weight() << endl;
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "gamma beta grad: " << endl;
    for (auto param : layernorm.parameters()) {
        cout << *param->get_grad() << endl;
    }
    cout << "x grad: " << endl;
    cout << *x->get_grad() << endl;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_softmax() {
    auto softmax_size = 6;
    Matrix *input = allocTmpMatrix(Shape(softmax_size, 2));
    for (int i = 0; i < 6; i++) {
        (*input)[i][0] = i;
        (*input)[i][1] = i+1;
    }
    std::vector<uint> labels = {2, 3};
    autograd::Node *x = autograd::allocNode(input);
    x->require_grad();

    autograd::Node *y = x->Softmax();
    auto loss = y->CrossEntropy(labels);

    loss->backward();
    // print y
    cout << "y: " << endl;
    cout << *y->get_weight() << endl;
    cout << "loss : " << endl;
    cout << *loss->get_weight() << endl;
    cout << "x grad: " << endl;
    cout << *x->get_grad() << endl;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention(const std::vector<uint> &valid_lens) {

    std::vector<autograd::Node *> queries;

    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;

    Matrix *mq2 = allocTmpMatrix(Shape(2, 1));
    (*mq2)[0][0] = 0.2;
    (*mq2)[1][0] = 0.2;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    queries.push_back(q1);
    queries.push_back(q2);

    std::vector<autograd::Node *> keys;

    Matrix *mk1 = allocTmpMatrix(Shape(2, 5));
    (*mk1)[0][0] = 1.1;
    (*mk1)[1][0] = 1.1;
    (*mk1)[0][1] = 1.2;
    (*mk1)[1][1] = 1.2;
    (*mk1)[0][2] = 1.3;
    (*mk1)[1][2] = 1.3;
    (*mk1)[0][3] = 1.4;
    (*mk1)[1][3] = 1.4;
    (*mk1)[0][4] = 1.5;
    (*mk1)[1][4] = 1.5;

    Matrix *mk2 = allocTmpMatrix(Shape(2, 5));
    (*mk2)[0][0] = 2.1;
    (*mk2)[1][0] = 2.1;
    (*mk2)[0][1] = 2.2;
    (*mk2)[1][1] = 2.2;
    (*mk2)[0][2] = 2.3;
    (*mk2)[1][2] = 2.3;
    (*mk2)[0][3] = 2.4;
    (*mk2)[1][3] = 2.4;
    (*mk2)[0][4] = 2.5;
    (*mk2)[1][4] = 2.5;

    autograd::Node *k1 = autograd::allocNode(mk1);
    autograd::Node *k2 = autograd::allocNode(mk2);

    k1->require_grad();
    k2->require_grad();

    keys.push_back(k1);
    keys.push_back(k2);

    std::vector<autograd::Node *> values;

    Matrix *mv1 = allocTmpMatrix(Shape(4, 5));
    (*mv1)[0][0] = 3.1;
    (*mv1)[1][0] = 3.1;
    (*mv1)[2][0] = 3.1;
    (*mv1)[3][0] = 3.1;
    (*mv1)[0][1] = 3.2;
    (*mv1)[1][1] = 3.2;
    (*mv1)[2][1] = 3.2;
    (*mv1)[3][1] = 3.2;
    (*mv1)[0][2] = 3.3;
    (*mv1)[1][2] = 3.3;
    (*mv1)[2][2] = 3.3;
    (*mv1)[3][2] = 3.3;
    (*mv1)[0][3] = 3.4;
    (*mv1)[1][3] = 3.4;
    (*mv1)[2][3] = 3.4;
    (*mv1)[3][3] = 3.4;
    (*mv1)[0][4] = 3.5;
    (*mv1)[1][4] = 3.5;
    (*mv1)[2][4] = 3.5;
    (*mv1)[3][4] = 3.5;

    Matrix *mv2 = allocTmpMatrix(Shape(4, 5));
    (*mv2)[0][0] = 4.1;
    (*mv2)[1][0] = 4.1;
    (*mv2)[2][0] = 4.1;
    (*mv2)[3][0] = 4.1;
    (*mv2)[0][1] = 4.2;
    (*mv2)[1][1] = 4.2;
    (*mv2)[2][1] = 4.2;
    (*mv2)[3][1] = 4.2;
    (*mv2)[0][2] = 4.3;
    (*mv2)[1][2] = 4.3;
    (*mv2)[2][2] = 4.3;
    (*mv2)[3][2] = 4.3;
    (*mv2)[0][3] = 4.4;
    (*mv2)[1][3] = 4.4;
    (*mv2)[2][3] = 4.4;
    (*mv2)[3][3] = 4.4;
    (*mv2)[0][4] = 4.5;
    (*mv2)[1][4] = 4.5;
    (*mv2)[2][4] = 4.5;
    (*mv2)[3][4] = 4.5;

    autograd::Node *v1 = autograd::allocNode(mv1);
    autograd::Node *v2 = autograd::allocNode(mv2);

    v1->require_grad();
    v2->require_grad();

    values.push_back(v1);
    values.push_back(v2);

    DotProductAttetion attention(0);

    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);

    std::vector<uint> labels = {2, 3};

    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);

    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "q1 grad: " << endl;
    cout << *q1->get_grad() << endl;
    cout << "q2 grad: " << endl;
    cout << *q2->get_grad() << endl;

    cout << "k1 grad: " << endl;
    cout << *k1->get_grad() << endl;
    cout << "k2 grad: " << endl;
    cout << *k2->get_grad() << endl;

    cout << "v1 grad: " << endl;
    cout << *v1->get_grad() << endl;
    cout << "v2 grad: " << endl;
    cout << *v2->get_grad() << endl;

    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention_without_mask() {
    std::vector<uint> valid_lens = {5, 5}; // all valid
    test_attention(valid_lens);
}

void test_attention_with_mask() {
    std::vector<uint> valid_lens = {2, 4}; // 3 valid for the first query
    test_attention(valid_lens);
}

void init_qkv_labels1(
    std::vector<autograd::Node *> &queries,
    std::vector<autograd::Node *> &keys,
    std::vector<autograd::Node *> &values,
    std::vector<uint> &labels
) {
    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;

    Matrix *mq2 = allocTmpMatrix(Shape(2, 1));
    (*mq2)[0][0] = 0.2;
    (*mq2)[1][0] = 0.2;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    queries.push_back(q1);
    queries.push_back(q2);

    Matrix *mk1 = allocTmpMatrix(Shape(2, 5));
    (*mk1)[0][0] = 1.1;
    (*mk1)[1][0] = 1.1;
    (*mk1)[0][1] = 1.2;
    (*mk1)[1][1] = 1.2;
    (*mk1)[0][2] = 1.3;
    (*mk1)[1][2] = 1.3;
    (*mk1)[0][3] = 1.4;
    (*mk1)[1][3] = 1.4;
    (*mk1)[0][4] = 1.5;
    (*mk1)[1][4] = 1.5;

    Matrix *mk2 = allocTmpMatrix(Shape(2, 5));
    (*mk2)[0][0] = 2.1;
    (*mk2)[1][0] = 2.1;
    (*mk2)[0][1] = 2.2;
    (*mk2)[1][1] = 2.2;
    (*mk2)[0][2] = 2.3;
    (*mk2)[1][2] = 2.3;
    (*mk2)[0][3] = 2.4;
    (*mk2)[1][3] = 2.4;
    (*mk2)[0][4] = 2.5;
    (*mk2)[1][4] = 2.5;

    autograd::Node *k1 = autograd::allocNode(mk1);
    autograd::Node *k2 = autograd::allocNode(mk2);

    k1->require_grad();
    k2->require_grad();

    keys.push_back(k1);
    keys.push_back(k2);

    Matrix *mv1 = allocTmpMatrix(Shape(4, 5));
    (*mv1)[0][0] = 3.1;
    (*mv1)[1][0] = 3.1;
    (*mv1)[2][0] = 3.1;
    (*mv1)[3][0] = 3.1;
    (*mv1)[0][1] = 3.2;
    (*mv1)[1][1] = 3.2;
    (*mv1)[2][1] = 3.2;
    (*mv1)[3][1] = 3.2;
    (*mv1)[0][2] = 3.3;
    (*mv1)[1][2] = 3.3;
    (*mv1)[2][2] = 3.3;
    (*mv1)[3][2] = 3.3;
    (*mv1)[0][3] = 3.4;
    (*mv1)[1][3] = 3.4;
    (*mv1)[2][3] = 3.4;
    (*mv1)[3][3] = 3.4;
    (*mv1)[0][4] = 3.5;
    (*mv1)[1][4] = 3.5;
    (*mv1)[2][4] = 3.5;
    (*mv1)[3][4] = 3.5;

    Matrix *mv2 = allocTmpMatrix(Shape(4, 5));
    (*mv2)[0][0] = 4.1;
    (*mv2)[1][0] = 4.1;
    (*mv2)[2][0] = 4.1;
    (*mv2)[3][0] = 4.1;
    (*mv2)[0][1] = 4.2;
    (*mv2)[1][1] = 4.2;
    (*mv2)[2][1] = 4.2;
    (*mv2)[3][1] = 4.2;
    (*mv2)[0][2] = 4.3;
    (*mv2)[1][2] = 4.3;
    (*mv2)[2][2] = 4.3;
    (*mv2)[3][2] = 4.3;
    (*mv2)[0][3] = 4.4;
    (*mv2)[1][3] = 4.4;
    (*mv2)[2][3] = 4.4;
    (*mv2)[3][3] = 4.4;
    (*mv2)[0][4] = 4.5;
    (*mv2)[1][4] = 4.5;
    (*mv2)[2][4] = 4.5;
    (*mv2)[3][4] = 4.5;

    autograd::Node *v1 = autograd::allocNode(mv1);
    autograd::Node *v2 = autograd::allocNode(mv2);

    v1->require_grad();
    v2->require_grad();

    values.push_back(v1);
    values.push_back(v2);

    labels.push_back(0);
    labels.push_back(0);
}

void init_qkv_labels0(
    std::vector<autograd::Node *> &queries,
    std::vector<autograd::Node *> &keys,
    std::vector<autograd::Node *> &values,
    std::vector<uint> &labels
) {
    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;

    Matrix *mq2 = allocTmpMatrix(Shape(2, 1));
    (*mq2)[0][0] = 0.2;
    (*mq2)[1][0] = 0.2;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    queries.push_back(q1);
    queries.push_back(q2);

    Matrix *mk1 = allocTmpMatrix(Shape(2, 2));
    (*mk1)[0][0] = 1.1;
    (*mk1)[1][0] = 1.1;
    (*mk1)[0][1] = 1.2;
    (*mk1)[1][1] = 1.2;

    Matrix *mk2 = allocTmpMatrix(Shape(2, 2));
    (*mk2)[0][0] = 2.1;
    (*mk2)[1][0] = 2.1;
    (*mk2)[0][1] = 2.1;
    (*mk2)[1][1] = 2.1;
    
    autograd::Node *k1 = autograd::allocNode(mk1);
    autograd::Node *k2 = autograd::allocNode(mk2);

    k1->require_grad();
    k2->require_grad();

    keys.push_back(k1);
    keys.push_back(k2);

    Matrix *mv1 = allocTmpMatrix(Shape(2, 2));
    (*mv1)[0][0] = 3.1;
    (*mv1)[1][0] = 3.1;
    (*mv1)[0][1] = 3.2;
    (*mv1)[1][1] = 3.5;

    Matrix *mv2 = allocTmpMatrix(Shape(2, 2));
    (*mv2)[0][0] = 4.1;
    (*mv2)[1][0] = 4.1;
    (*mv2)[0][1] = 4.1;
    (*mv2)[1][1] = 4.1;
    
    autograd::Node *v1 = autograd::allocNode(mv1);
    autograd::Node *v2 = autograd::allocNode(mv2);

    v1->require_grad();
    v2->require_grad();

    values.push_back(v1);
    values.push_back(v2);

    labels.push_back(0);
    labels.push_back(0);
}

void print_qkv_res_grad(
    const std::vector<autograd::Node *> &queries,
    const std::vector<autograd::Node *> &keys,
    const std::vector<autograd::Node *> &values,
    const std::vector<autograd::Node *> &res
) {
    cout << "res : " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    cout << "q1 grad: " << endl;
    cout << *queries[0]->get_grad() << endl;
    cout << "q2 grad: " << endl;
    cout << *queries[1]->get_grad() << endl;

    cout << "k1 grad: " << endl;
    cout << *keys[0]->get_grad() << endl;
    cout << "k2 grad: " << endl;
    cout << *keys[1]->get_grad() << endl;

    cout << "v1 grad: " << endl;
    cout << *values[0]->get_grad() << endl;
    cout << "v2 grad: " << endl;
    cout << *values[1]->get_grad() << endl;

    cout << "res grad: " << endl;
    for (auto r : res) {
        cout << *r->get_grad() << endl;
    }
}

void test_mh_attention(
    const std::vector<uint> &valid_lens,
    std::vector<autograd::Node *> queries,
    std::vector<autograd::Node *> keys,
    std::vector<autograd::Node *> values,
    std::vector<uint> labels,
    uint num_hidden,
    uint num_heads) {
    
    MultiHeadAttention attention(num_heads, num_hidden, 0);
    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    print_qkv_res_grad(queries, keys, values, res);
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_mh_attention_2d(
    const std::vector<std::vector<uint>> &valid_lens,
    std::vector<autograd::Node *> queries,
    std::vector<autograd::Node *> keys,
    std::vector<autograd::Node *> values,
    std::vector<uint> labels,
    uint num_hidden,
    uint num_heads) {
    
    MultiHeadAttention attention(num_heads, num_hidden, 0);
    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    print_qkv_res_grad(queries, keys, values, res);
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention1(const std::vector<uint> &valid_lens) {
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels0(queries, keys, values, labels);
    DotProductAttetion attention(0);
    std::vector<autograd::Node *> res = attention.forward(queries, keys, values, valid_lens);
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    print_qkv_res_grad(queries, keys, values, res);
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_attention_to_cp_with_mha() {
    std::vector<uint> valid_lens = {5, 5}; // all valid
    test_attention1(valid_lens);
}

void test_mh_attention_without_mask0() {
    std::vector<uint> valid_lens = {5, 5}; // all valid
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels0(queries, keys, values, labels);
    test_mh_attention(valid_lens, queries, keys, values, labels, 3);
}

void test_mh_attention_without_mask1() {
    std::vector<uint> valid_lens = {5, 5}; // all valid
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels1(queries, keys, values, labels);
    test_mh_attention(valid_lens, queries, keys, values, labels, 10);
}

void test_mh_attention_without_mask2() {
    std::vector<uint> valid_lens = {}; // all valid
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels1(queries, keys, values, labels);
    test_mh_attention(valid_lens, queries, keys, values, labels, 10);
}

void test_mh_attention_with_mask() {
    std::vector<uint> valid_lens = {2, 4};
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels1(queries, keys, values, labels);
    test_mh_attention(valid_lens, queries, keys, values, labels, 10, 2);
}

void test_lazy_liner() {
    std::vector<autograd::Node *> queries;
    Matrix *mq1 = allocTmpMatrix(Shape(2, 1));
    (*mq1)[0][0] = 0.1;
    (*mq1)[1][0] = 0.1;
    autograd::Node *q1 = autograd::allocNode(mq1);
    q1->require_grad();
    auto W = new autograd::LazyLinear(3, autograd::ACTIVATION::NONE, false);
    std::vector<autograd::Node *> res = {W->forward(q1)};
    std::vector<uint> labels = {2};
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy(labels);
    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    cout << "q1 grad: " << endl;
    cout << *q1->get_grad() << endl;
    delete W;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();  
}

void test_pos_encoding() {
    // uint num_hidden = 256;
    uint num_hidden = 20;
    PosEncoding *pos_encoding = new PosEncoding(2, num_hidden, 0);
    Matrix *input0 = allocTmpMatrix(Shape(num_hidden, 2));
    Matrix *input1 = allocTmpMatrix(Shape(num_hidden, 2));
    input1->fill(1);
    std::vector<autograd::Node *> x;
    autograd::Node *x0 = autograd::allocNode(input0);
    autograd::Node *x1 = autograd::allocNode(input1);
    x0->require_grad();
    x1->require_grad();
    x.push_back(x0);
    x.push_back(x1);
    std::vector<autograd::Node *> res = pos_encoding->forward(x);
    // print res
    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }
    delete pos_encoding;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_addnorm() {
    Matrix *mq1 = allocTmpMatrix(Shape(4, 1));
    (*mq1)[0][0] = 3.5;
    (*mq1)[1][0] = 3.1;
    (*mq1)[2][0] = 3.1;
    (*mq1)[3][0] = 3.1;

    Matrix *mq2 = allocTmpMatrix(Shape(4, 1));
    (*mq2)[0][0] = 4.5;
    (*mq2)[1][0] = 4.1;
    (*mq2)[2][0] = 4.1;
    (*mq2)[3][0] = 4.1;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    std::vector<autograd::Node *> queries = {q1, q2};
    AddNorm *addnorm = new AddNorm(4, 0);
    std::vector<autograd::Parameters *> params = addnorm->get_parameters();
    //std::vector<autograd::Node *> res = addnorm->forward(x, x);
    std::vector<autograd::Node *> res;
    for (auto q : queries) {
        // print q
        cout << "q: " << endl;
        cout << *q->get_weight() << endl;
        res.push_back(addnorm->forward(q, q));
    }

    // print res
    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy({0, 0});
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    cout << "res grad: " << endl;
    for (auto r : res) {
        cout << *r->get_grad() << endl;
    }

    cout << "q1 grad: " << endl;
    cout << *q1->get_grad() << endl;
    cout << "q2 grad: " << endl;
    cout << *q2->get_grad() << endl;
    cout << "gamma grad: " << endl;
    cout << *params[0]->get_grad() << endl;
    cout << "beta grad: " << endl;
    cout << *params[1]->get_grad() << endl;
    delete addnorm;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_ffn() {
    Matrix *mq1 = allocTmpMatrix(Shape(4, 1));
    (*mq1)[0][0] = 3.5;
    (*mq1)[1][0] = 3.1;
    (*mq1)[2][0] = 3.1;
    (*mq1)[3][0] = 3.1;

    Matrix *mq2 = allocTmpMatrix(Shape(4, 1));
    (*mq2)[0][0] = 4.5;
    (*mq2)[1][0] = 4.1;
    (*mq2)[2][0] = 4.1;
    (*mq2)[3][0] = 4.1;

    autograd::Node *q1 = autograd::allocNode(mq1);
    autograd::Node *q2 = autograd::allocNode(mq2);

    q1->require_grad();
    q2->require_grad();

    std::vector<autograd::Node *> queries = {q1, q2};
    PositionwiseFFN *ffn = new PositionwiseFFN(8, 5);
    std::vector<autograd::Node *> res;

    for (auto q : queries) {
        res.push_back(ffn->forward(q));
    }

    // print res
    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy({0, 0});
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    cout << "res grad: " << endl;
    for (auto r : res) {
        cout << *r->get_grad() << endl;
    }

    cout << "q1 grad: " << endl;
    cout << *q1->get_grad() << endl;
    cout << "q2 grad: " << endl;
    cout << *q2->get_grad() << endl;
    delete ffn;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_encoder() {
    std::vector<std::vector<uint>> inputs;
    inputs.push_back({0, 1, 2});
    inputs.push_back({0, 2, 3});

    uint num_hiddens = 16;
    uint num_blks = 2;
    float dropout = 0;
    uint ffn_num_hiddens = 4;
    uint num_heads = 4;
    uint vocab_size = 4;

    Encoder *encoder = new Encoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout);
    std::vector<autograd::Node *> embs;
    std::vector<autograd::Node *> res = encoder->forward(inputs, {}, embs);

    // print res
    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }

    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy({0, 0, 0, 0, 0, 0});

    cout << "loss: " << endl;

    cout << *loss->get_weight() << endl;

    loss->backward();

    cout << "res grad: " << endl;

    for (auto r : res) {
        cout << *r->get_grad() << endl;
    }

    cout << "embs:" << endl;
    for (auto e : embs) {
        cout << *e->get_weight() << endl;
    }

    cout << "embs grad:" << endl;
    for (auto e : embs) {
        cout << *e->get_grad() << endl;
    }

    delete encoder;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test_mh_attention_with_2d_mask() {
    std::vector<std::vector<uint>> valid_lens;
    valid_lens.push_back({2});
    valid_lens.push_back({4});
    std::vector<autograd::Node *> queries;
    std::vector<autograd::Node *> keys;
    std::vector<autograd::Node *> values;
    std::vector<uint> labels;
    init_qkv_labels1(queries, keys, values, labels);
    test_mh_attention_2d(valid_lens, queries, keys, values, labels, 10, 2);
}

void test_decoder() {


    std::vector<std::vector<uint>> inputs;
    inputs.push_back({0, 1, 2});
    inputs.push_back({0, 2, 3});

    uint num_hiddens = 16;
    uint num_blks = 2;
    float dropout = 0;
    uint ffn_num_hiddens = 4;
    uint num_heads = 4;
    uint vocab_size = 4;

    Decoder *decoder = new Decoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout);
    std::vector<autograd::Node *> embs;
    std::vector<uint> enc_valid_lens = {};
    std::vector<autograd::Node *> enc_output;
    Matrix *menc_o1 = allocTmpMatrix(Shape(3, 2));
    menc_o1->fill(1);
    Matrix *menc_o2 = allocTmpMatrix(Shape(3, 2));
    menc_o2->fill(1);
    autograd::Node *enc_o1 = autograd::allocNode(menc_o1);
    autograd::Node *enc_o2 = autograd::allocNode(menc_o2);
    enc_o1->require_grad();
    enc_o2->require_grad();
    enc_output.push_back(enc_o1);
    enc_output.push_back(enc_o2);
    std::vector<autograd::Node *> res = decoder->forward(inputs, enc_output, enc_valid_lens, embs);
    // print res
    cout << "res: " << endl;
    for (auto r : res) {
        cout << *r->get_weight() << endl;
    }
    autograd::Node *loss = autograd::cat(res, 0)->CrossEntropy({0, 0, 0, 0, 0, 0});
    cout << "loss: " << endl;
    cout << *loss->get_weight() << endl;
    loss->backward();
    cout << "res grad: " << endl;
    for (auto r : res) {
        cout << *r->get_grad() << endl;
    }
    cout << "embs:" << endl;
    for (auto e : embs) {
        cout << *e->get_weight() << endl;
    }
    cout << "embs grad:" << endl;
    for (auto e : embs) {
        cout << *e->get_grad() << endl;
    }
    delete decoder;
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
}

void test() {
    // test_layernorm();
    // test_softmax();
    // test_attention_without_mask();
    // test_attention_with_mask();
    // cout << "------ test_mh_attention_without_mask0 ------" << endl;
    // test_mh_attention_without_mask0();
    // cout << "------ test_mh_attention_without_mask0 end ------" << endl;

    // cout << "------ test_mh_attention_without_mask1 ------" << endl;
    // test_mh_attention_without_mask1();
    // cout << "------ test_mh_attention_without_mask1 end ------" << endl;
    // cout << "------ test_attention_to_cp_with_mha ------" << endl;
    // test_attention_to_cp_with_mha();
    // cout << "------ test_attention_to_cp_with_mha end ------" << endl;
    // test_lazy_liner();
    // test_mh_attention_with_mask();
    test_pos_encoding();
    // test_addnorm();
    // test_ffn();
    // test_mh_attention_without_mask1();
    // test_mh_attention_without_mask2();
    // test_encoder();

    // test_mh_attention_with_2d_mask();
    // test_decoder();
}