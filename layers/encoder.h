#ifndef ENCODER_H
#define ENCODER_H

#include "autograd/node.h"
#include "dropout.h"
#include "attention.h"
#include "addnorm.h"
#include "ffn.h"
#include "embedding.h"
#include "posencoding.h"

class EncoderBlock {
    public:
        EncoderBlock(uint _num_hidden, uint _ffn_num_hiddens, uint _num_heads, DATATYPE dropout, bool _bias = false);
        ~EncoderBlock();
        std::vector<autograd::Node *> forward(const std::vector<autograd::Node *> &x, const std::vector<uint> &valid_lens);
        std::vector<autograd::Parameters *> get_parameters();
        void train(bool _training);
        bool is_training();
    private:
        uint num_hidden;
        uint ffn_num_hiddens;
        uint num_heads;
        bool training;
        bool bias;
        MultiHeadAttention *attention;
        PositionwiseFFN *ffn;
        AddNorm *addnorm1;
        AddNorm *addnorm2;
};

class Encoder {
    public:
        Encoder(
            uint _vocab_size,
            uint _num_hidden,
            uint _ffn_num_hiddens,
            uint _num_heads,
            uint _num_blocks,
            DATATYPE dropout,
            bool _bias = false
        );
        ~Encoder();
        std::vector<autograd::Node *> forward(
            const std::vector<std::vector<uint>> &inputs,
            const std::vector<uint> &valid_lens,
            std::vector<autograd::Node *> &out_embs
        );
        std::vector<autograd::Parameters *> get_parameters();
        void train(bool _training);
        bool is_training();
    private:
        uint vocab_size;
        uint num_hidden;
        uint ffn_num_hiddens;
        uint num_heads;
        uint num_blocks;
        bool training;
        bool bias;
        std::vector<EncoderBlock *> blocks;
        autograd::Embedding *embedding;
        PosEncoding *posencoding;
};

#endif