#include "transformer_encoder.h"
#include <cmath>
#include <sstream>

TransformerEncoder::TransformerEncoder(
    int vocab_size, int _num_hiddens, int ffn_num_hiddens,
    int num_heads, int num_blks, int max_posencoding_len,
    float dropout, bool bias
) : num_hiddens(_num_hiddens) {
    embedding = new Embedding(vocab_size, num_hiddens);
    pos_encoding = new PosEncoding(max_posencoding_len, num_hiddens, dropout);
    for (int i = 0; i < num_blks; i++) {
        blks.push_back(new TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, bias));
    }
}

TransformerEncoder::~TransformerEncoder() {
    delete embedding;
    delete pos_encoding;
    for (auto blk : blks) {
        delete blk;
    }
}

graph::Node* TransformerEncoder::forward(Tensor* indices, Tensor* valid_lens) {
    assert(indices->get_dim() == 2); // shape : (batch_size, seq_len)
    auto indices_shape = indices->get_shape();
    auto x = embedding->forward(indices);
    x = x->mulsv(std::sqrt(num_hiddens));
    auto x_shape = x->get_tensor()->get_shape();
    assert(x_shape[0] == indices_shape[0]);
    assert(x_shape[1] == indices_shape[1]);
    x = pos_encoding->forward(x);
    for (auto blk : blks) {
        x = blk->forward(x, valid_lens);
    }
    return x;
}

std::vector<Parameter*> TransformerEncoder::get_parameters() {
    std::vector<Parameter*> parameters;
    auto embedding_params = embedding->get_parameters();
    parameters.insert(parameters.end(), embedding_params.begin(), embedding_params.end());
    for (auto blk : blks) {
        auto blk_params = blk->get_parameters();
        parameters.insert(parameters.end(), blk_params.begin(), blk_params.end());
    }
    return parameters;
}