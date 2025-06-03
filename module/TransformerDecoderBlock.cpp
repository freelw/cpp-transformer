#include "TransformerDecoderBlock.h"

TransformerDecoderBlock::TransformerDecoderBlock(
    int num_hiddens, int ffn_num_hiddens, int num_heads,
    float dropout, bool bias) {
    masked_attention = new MHA(
        num_hiddens, num_heads, dropout, bias
    );
    addnorm1 = new AddNorm(
        num_hiddens, dropout
    );
    attention = new MHA(
        num_hiddens, num_heads, dropout, bias
    );
    addnorm2 = new AddNorm(
        num_hiddens, dropout
    );
    ffn = new PositionWiseFFN(
        ffn_num_hiddens, num_hiddens
    );
    addnorm3 = new AddNorm(
        num_hiddens, dropout
    );
}

TransformerDecoderBlock::~TransformerDecoderBlock() {
    delete masked_attention;
    delete addnorm1;
    delete attention;
    delete addnorm2;
    delete ffn;
    delete addnorm3;
}

graph::Node* TransformerDecoderBlock::forward(
    graph::Node* x, graph::Node* enc_output,
    Tensor* enc_valid_lens, Tensor* dec_valid_lens) {
    auto y = masked_attention->forward(x, x, x, dec_valid_lens);
    y = addnorm1->forward(x, y);
    auto z = attention->forward(y, enc_output, enc_output, enc_valid_lens);
    z = addnorm2->forward(y, z);
    auto out = ffn->forward(z);
    auto res = addnorm3->forward(z, out);
    return res;
}

std::vector<Parameter*> TransformerDecoderBlock::get_parameters() {
    std::vector<Parameter*> params;
    auto masked_attention_params = masked_attention->get_parameters();
    auto addnorm1_params = addnorm1->get_parameters();
    auto attention_params = attention->get_parameters();
    auto addnorm2_params = addnorm2->get_parameters();
    auto ffn_params = ffn->get_parameters();
    auto addnorm3_params = addnorm3->get_parameters();

    params.insert(params.end(), masked_attention_params.begin(), masked_attention_params.end());
    params.insert(params.end(), addnorm1_params.begin(), addnorm1_params.end());
    params.insert(params.end(), attention_params.begin(), attention_params.end());
    params.insert(params.end(), addnorm2_params.begin(), addnorm2_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    params.insert(params.end(), addnorm3_params.begin(), addnorm3_params.end());

    return params;
}