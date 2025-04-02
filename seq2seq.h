#ifndef TRANSFORMER_SEQ2SEQ_H
#define TRANSFORMER_SEQ2SEQ_H

#include "encoder.h"
#include "decoder.h"
class Seq2SeqEncoderDecoder {
    public:
        Seq2SeqEncoderDecoder(
            Encoder *_encoder,
            Decoder *_decoder,
            uint _bos_id,
            uint _eos_id
        ) : encoder(_encoder), decoder(_decoder),
            bos_id(_bos_id), eos_id(_eos_id) {}
        ~Seq2SeqEncoderDecoder() {}
        autograd::Node* forward(
            const std::vector<std::vector<uint>> &src_token_ids,
            const std::vector<std::vector<uint>> &tgt_token_ids,
            const std::vector<uint> &valid_lens,
            std::vector<autograd::Node *> &out_embs,
            std::vector<autograd::Node *> &dec_out_embs
        );
        std::vector<autograd::Parameters *> get_parameters();
        void train(bool _training) {
            encoder->train(_training);
            decoder->train(_training);
        }
        std::vector<uint> predict(
            const std::vector<uint> &src_token_ids,
            uint max_len,
            std::vector<autograd::Node *> &enc_out_embs,
            std::vector<autograd::Node *> &dec_out_embs
        );
        Encoder *get_encoder() { return encoder; }
        Decoder *get_decoder() { return decoder; }
    private:
        Encoder *encoder;
        Decoder *decoder;
        uint bos_id;
        uint eos_id;
};

#endif