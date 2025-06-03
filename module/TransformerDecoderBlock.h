#ifndef TRANSFORMERDECODERBLOCK_H
#define TRANSFORMERDECODERBLOCK_H

#include "module/mha.h"
#include "module/PositionWiseFFN.h"
#include "module/AddNorm.h"

class TransformerDecoderBlock {
public:
    TransformerDecoderBlock(int num_hiddens, int ffn_num_hiddens, int num_heads,
        float dropout, bool bias = false);
    ~TransformerDecoderBlock();
    graph::Node* forward(graph::Node* x, graph::Node* enc_output, Tensor* enc_valid_lens, Tensor* dec_valid_lens = nullptr);
    std::vector<Parameter*> get_parameters();
private:
    MHA* masked_attention;
    AddNorm* addnorm1;
    MHA* attention; // encoder decoder attention
    AddNorm* addnorm2;
    PositionWiseFFN* ffn;
    AddNorm* addnorm3;
};

#endif