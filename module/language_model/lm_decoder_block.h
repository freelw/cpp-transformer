#ifndef LM_DECODER_BLOCK_H
#define LM_DECODER_BLOCK_H

#include "module/mha.h"
#include "module/position_wise_ffn.h"
#include "module/addnorm.h"

class LMDecoderBlock {
public:
    LMDecoderBlock(int num_hiddens, int ffn_num_hiddens, int num_heads,
        float dropout, bool bias = false);
    ~LMDecoderBlock();
    graph::Node* forward(graph::Node* x, Tensor* dec_valid_lens = nullptr);
    std::vector<Parameter*> get_parameters();
private:
    MHA* masked_attention;
    AddNorm* addnorm1;
    PositionWiseFFN* ffn;
    AddNorm* addnorm2;
};
#endif