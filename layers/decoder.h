#ifndef DECODER_H
#define DECODER_H


#include "autograd/node.h"
#include "attention.h"
#include "addnorm.h"
#include "ffn.h"
#include "embedding.h"
#include "posencoding.h"

struct DecoderContext {
    std::vector<autograd::Node *> dec_X;
};

class DecoderBlock {
    public:
        DecoderBlock(
            uint _num_hidden,
            uint _ffn_num_hiddens,
            uint _num_heads,
            DATATYPE dropout,
            uint _index
        );
        ~DecoderBlock();
        bool is_training() { return training; }
        void train(bool _training);
        std::vector<autograd::Node *> forward(
            const std::vector<autograd::Node *> &X,
            const std::vector<autograd::Node *> &enc_outputs,
            const std::vector<uint> &enc_valid_lens,
            DecoderContext *ctx
        );
        std::vector<autograd::Parameters *> get_parameters();
    private:
        uint num_hidden;
        uint ffn_num_hiddens;
        uint num_heads;
        uint index;
        bool training;
        MultiHeadAttention *self_attention;
        MultiHeadAttention *enc_attention;
        PositionwiseFFN *ffn;
        AddNorm *addnorm1;
        AddNorm *addnorm2;
        AddNorm *addnorm3;
};

class Decoder {
    public:
        Decoder(
            uint _vocab_size,
            uint _num_hidden,
            uint _ffn_num_hiddens,
            uint _num_heads,
            uint _num_blks,
            DATATYPE dropout
        );
        ~Decoder();
        std::vector<autograd::Node *> forward(
            const std::vector<std::vector<uint>> &inputs,
            const std::vector<autograd::Node *> &enc_outputs,
            const std::vector<uint> &enc_valid_lens,
            std::vector<autograd::Node *> &out_embs
        );
        void train(bool _training);
        bool is_training() { return training; }
        std::vector<autograd::Parameters *> get_parameters();
    private:
        uint vocab_size;
        uint num_hidden;
        uint ffn_num_hiddens;
        uint num_heads;
        uint num_blks;
        bool training;
        autograd::Embedding *embedding;
        PosEncoding *posencoding;
        autograd::LazyLinear *dense;
        std::vector<DecoderBlock *> blocks;
};
#endif