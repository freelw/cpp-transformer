#include "TransformerDecoder.h"
#include <cmath>

TransformerDecoder::TransformerDecoder(
    int vocab_size,
    int _num_hiddens,
    int ffn_num_hiddens,
    int num_heads,
    int num_blks,
    int max_posencoding_len,
    float dropout,
    bool bias
) : num_hiddens(_num_hiddens) {
    embedding = new Embedding(vocab_size, num_hiddens);
    pos_encoding = new PosEncoding(max_posencoding_len, num_hiddens, dropout);
    
    for (int i = 0; i < num_blks; i++) {
        blks.push_back(new TransformerDecoderBlock(
            num_hiddens, ffn_num_hiddens, num_heads, dropout, bias
        ));
    }
    dense = new LazyLinear(vocab_size, "dense", -1.0f, -1.0f, NONE);
}

TransformerDecoder::~TransformerDecoder() {
    assert(embedding != nullptr);
    assert(pos_encoding != nullptr);
    assert(dense != nullptr);
    for (auto blk : blks) {
        assert(blk != nullptr);
        delete blk;
    }
    delete embedding;
    delete pos_encoding;
    delete dense;
}

graph::Node *TransformerDecoder::forward(
    Tensor *tgt_token_ids,
    graph::Node *enc_outputs,
    Tensor *enc_valid_lens,
    Tensor *dec_valid_lens
) {
    auto x = embedding->forward(tgt_token_ids)->mulsv(std::sqrt(num_hiddens));
    x = pos_encoding->forward(x);
    for (auto blk : blks) {
        x = blk->forward(x, enc_outputs, enc_valid_lens, dec_valid_lens);
    }
    auto dense_res = dense->forward(x);
    return dense_res;
}

std::vector<Parameter *> TransformerDecoder::get_parameters() {
    std::vector<Parameter *> parameters;
    auto embedding_params = embedding->get_parameters();
    parameters.insert(parameters.end(), embedding_params.begin(), embedding_params.end());
    for (auto blk : blks) {
        auto blk_params = blk->get_parameters();
        parameters.insert(parameters.end(), blk_params.begin(), blk_params.end());
    }
    auto dense_params = dense->get_parameters();
    parameters.insert(parameters.end(), dense_params.begin(), dense_params.end());
    return parameters;
}
