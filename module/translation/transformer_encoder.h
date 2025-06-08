#ifndef TRANSFORMERENCODER_H
#define TRANSFORMERENCODER_H

#include "module/translation/transformer_encoder_block.h"
#include "module/embedding.h"
#include "module/pos_encoding.h"

class TransformerEncoder {
public:
    TransformerEncoder(
        int vocab_size, int num_hiddens, int ffn_num_hiddens,
        int num_heads, int num_blks, int max_posencoding_len,
        float dropout, bool bias = false
    );
    ~TransformerEncoder();
    graph::Node* forward(Tensor* indices, Tensor* valid_lens = nullptr);
    std::vector<Parameter*> get_parameters();
private:
    int num_hiddens;
    Embedding* embedding;
    PosEncoding* pos_encoding;
    std::vector<TransformerEncoderBlock*> blks;
};


#endif