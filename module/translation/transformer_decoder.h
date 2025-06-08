#ifndef TRANSFORMERDECODER_H
#define TRANSFORMERDECODER_H

#include "module/embedding.h"
#include "module/pos_encoding.h"
#include "module/linear.h"
#include "module/translation/transformer_decoder_block.h"

class TransformerDecoder {
public:
    TransformerDecoder(
        int vocab_size,
        int _num_hiddens,
        int ffn_num_hiddens,
        int num_heads,
        int num_blks,
        int max_posencoding_len,
        float dropout = 0.0f,
        bool bias = false
    );
    ~TransformerDecoder();
    graph::Node* forward(
        Tensor* tgt_token_ids,
        graph::Node* enc_outputs,
        Tensor* enc_valid_lens = nullptr,
        Tensor* dec_valid_lens = nullptr
    );
    std::vector<Parameter*> get_parameters();

private:
    int num_hiddens;
    Embedding* embedding;
    PosEncoding* pos_encoding;
    std::vector<TransformerDecoderBlock*> blks;
    LazyLinear* dense;
};

#endif