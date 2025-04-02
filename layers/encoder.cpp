#include "encoder.h"
#include "macro.h"

EncoderBlock::EncoderBlock(uint _num_hidden, uint _ffn_num_hiddens, uint _num_heads, DATATYPE dropout, bool _bias)
    : num_hidden(_num_hidden),
    ffn_num_hiddens(_ffn_num_hiddens),
    num_heads(_num_heads),
    training(true),
    bias(_bias)
{
    attention = new MultiHeadAttention(num_heads, num_hidden, dropout, bias);
    ffn = new PositionwiseFFN(_ffn_num_hiddens, num_hidden);
    addnorm1 = new AddNorm(num_hidden, dropout);
    addnorm2 = new AddNorm(num_hidden, dropout);
}

EncoderBlock::~EncoderBlock() {
    delete attention;
    delete ffn;
    delete addnorm1;
    delete addnorm2;
}

std::vector<autograd::Node *> EncoderBlock::forward(
    const std::vector<autograd::Node *> &x,
    const std::vector<uint> &valid_lens) {
    std::vector<autograd::Node *> y = addnorm1->forward(x, attention->forward(x, x, x, valid_lens));
    return addnorm2->forward(y, ffn->forward(y));
}

std::vector<autograd::Parameters *> EncoderBlock::get_parameters() {
    std::vector<autograd::Parameters *> res;
    auto p1 = attention->get_parameters();
    auto p2 = ffn->get_parameters();
    auto p3 = addnorm1->get_parameters();
    auto p4 = addnorm2->get_parameters();
    res.insert(res.end(), p1.begin(), p1.end());
    res.insert(res.end(), p2.begin(), p2.end());
    res.insert(res.end(), p3.begin(), p3.end());
    res.insert(res.end(), p4.begin(), p4.end());
    return res;
}

void EncoderBlock::train(bool _training) {
    training = _training;
    attention->train(_training);
    addnorm1->train(_training);
    addnorm2->train(_training);
}

bool EncoderBlock::is_training() {
    return training;
}

Encoder::Encoder(
    uint _vocab_size,
    uint _num_hidden,
    uint _ffn_num_hiddens,
    uint _num_heads,
    uint _num_blocks,
    DATATYPE dropout,
    bool _bias)
    : vocab_size(_vocab_size),
    num_hidden(_num_hidden),
    ffn_num_hiddens(_ffn_num_hiddens),
    num_heads(_num_heads),
    num_blocks(_num_blocks),
    training(true),
    bias(_bias)
{
    embedding = new autograd::Embedding(vocab_size, num_hidden);
    posencoding = new PosEncoding(MAX_POSENCODING_LEN, num_hidden, dropout);
    for (uint i = 0; i < num_blocks; i++) {
        blocks.push_back(new EncoderBlock(num_hidden, ffn_num_hiddens, num_heads, dropout, bias));
    }
}

Encoder::~Encoder() {
    for (auto & block : blocks) {
        delete block;
    }
    delete posencoding;
    delete embedding;
}

std::vector<autograd::Node *> Encoder::forward(
    const std::vector<std::vector<uint>> &inputs,
    const std::vector<uint> &valid_lens,
    std::vector<autograd::Node *> &out_embs) {
    auto embs = embedding->forward(inputs);
    out_embs = embs;
    std::vector<autograd::Node *> X;
    X.reserve(embs.size());
    for (auto & emb : embs) {
        auto p = emb->Mul(sqrt(num_hidden));
        assert(p->is_require_grad());
        X.push_back(p);
    }
    X = posencoding->forward(X);
    for (auto blk : blocks) {
        X = blk->forward(X, valid_lens);
    }
    return X;
}

std::vector<autograd::Parameters *> Encoder::get_parameters() {
    std::vector<autograd::Parameters *> res;
    for (auto & block : blocks) {
        auto p = block->get_parameters();
        res.insert(res.end(), p.begin(), p.end());
    }
    auto p = embedding->get_parameters();
    res.insert(res.end(), p.begin(), p.end());
    return res;
}

void Encoder::train(bool _training) {
    training = _training;
    for (auto & block : blocks) {
        block->train(_training);
    }
    posencoding->train(_training);
}

bool Encoder::is_training() {
    return training;
}