#ifndef TRANSFORMER_SEQ2SEQ_H
#define TRANSFORMER_SEQ2SEQ_H
#include "module/transformer_encoder.h"
#include "module/transformer_decoder.h"

class Seq2SeqEncoderDecoder {
public:
    Seq2SeqEncoderDecoder(
        uint _bos_id,
        uint _eos_id,
        int enc_vocab_size, int dec_vocab_size, int num_hiddens, int ffn_num_hiddens,
        int num_heads, int num_blks, int max_posencoding_len,
        float dropout, bool bias = false
    );
    ~Seq2SeqEncoderDecoder();
    graph::Node* forward(
        Tensor* src_token_ids,
        Tensor* tgt_token_ids,
        Tensor* enc_valid_lens,
        Tensor* dec_valid_lens
    );
    std::vector<Parameter*> get_parameters();
    TransformerEncoder* get_encoder() const {
        return encoder;
    }
    TransformerDecoder* get_decoder() const {
        return decoder;
    }
private:
    TransformerEncoder* encoder;
    TransformerDecoder* decoder;
    uint bos_id;
    uint eos_id;
};

#endif