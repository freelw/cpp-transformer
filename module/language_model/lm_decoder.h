#ifndef LM_DECODER_H
#define LM_DECODER_H

#include "module/embedding.h"
#include "module/pos_encoding.h"
#include "module/linear.h"
#include "module/language_model/lm_decoder_block.h"

class LMDecoder {
public:
    LMDecoder(
        int vocab_size,
        int _num_hiddens,
        int ffn_num_hiddens,
        int num_heads,
        int num_blks,
        int max_posencoding_len,
        float dropout = 0.0f,
        bool bias = false
    );
    ~LMDecoder();
    graph::Node* forward(
        Tensor* tgt_token_ids,
        Tensor* dec_valid_lens = nullptr
    );
    std::vector<Parameter*> get_parameters();

private:
    int num_hiddens;
    Embedding* embedding;
    PosEncoding* pos_encoding;
    std::vector<LMDecoderBlock*> blks;
    LazyLinear* dense;
};
#endif