#include "lm_decoder_block.h"

LMDecoderBlock::LMDecoderBlock(
    int num_hiddens, int ffn_num_hiddens, int num_heads,
    float dropout, bool bias) {
    masked_attention = new MHA(
        num_hiddens, num_heads, dropout, bias
    );
    addnorm1 = new AddNorm(
        num_hiddens, dropout
    );
    addnorm2 = new AddNorm(
        num_hiddens, dropout
    );
    ffn = new PositionWiseFFN(
        ffn_num_hiddens, num_hiddens
    );
}

LMDecoderBlock::~LMDecoderBlock() {
    delete masked_attention;
    delete addnorm1;
    delete addnorm2;
    delete ffn;
}

graph::Node* LMDecoderBlock::forward(
    graph::Node* x, Tensor* dec_valid_lens) {
    auto y = masked_attention->forward(x, x, x, dec_valid_lens);
    y = addnorm1->forward(x, y);
    auto out = ffn->forward(y);
    auto res = addnorm2->forward(y, out);
    return res;
}

std::vector<Parameter*> LMDecoderBlock::get_parameters() {
    std::vector<Parameter*> params;
    auto masked_attention_params = masked_attention->get_parameters();
    auto addnorm1_params = addnorm1->get_parameters();
    auto addnorm2_params = addnorm2->get_parameters();
    auto ffn_params = ffn->get_parameters();

    params.insert(params.end(), masked_attention_params.begin(), masked_attention_params.end());
    params.insert(params.end(), addnorm1_params.begin(), addnorm1_params.end());
    params.insert(params.end(), addnorm2_params.begin(), addnorm2_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());

    return params;
}