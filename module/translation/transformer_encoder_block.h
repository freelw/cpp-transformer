#ifndef TRANSFORMERENCODERBLOCK_H
#define TRANSFORMERENCODERBLOCK_H

#include "module/mha.h"
#include "module/position_wise_ffn.h"
#include "module/addnorm.h"

class TransformerEncoderBlock {
public:
    TransformerEncoderBlock(
        int num_hiddens, int ffn_num_hiddens, int num_heads,
        float dropout, bool bias = false
    );
    ~TransformerEncoderBlock();
    graph::Node* forward(graph::Node* x, Tensor* valid_lens = nullptr);
    std::vector<Parameter*> get_parameters();
private:
    MHA* attention;
    AddNorm* addnorm1;
    PositionWiseFFN* ffn;
    AddNorm* addnorm2;
};

#endif