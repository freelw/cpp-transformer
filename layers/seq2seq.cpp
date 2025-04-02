#include "seq2seq.h"

autograd::Node* Seq2SeqEncoderDecoder::forward(
    const std::vector<std::vector<uint>> &src_token_ids,
    const std::vector<std::vector<uint>> &tgt_token_ids,
    const std::vector<uint> &valid_lens,
    std::vector<autograd::Node *> &enc_out_embs,
    std::vector<autograd::Node *> &dec_out_embs
) {
    auto hiddens = encoder->forward(src_token_ids, valid_lens, enc_out_embs);
    auto dec_outputs = decoder->forward(tgt_token_ids, hiddens, valid_lens, dec_out_embs);
    return autograd::cat(dec_outputs);
}

std::vector<autograd::Parameters *> Seq2SeqEncoderDecoder::get_parameters() {
    auto encoder_params = encoder->get_parameters();
    auto decoder_params = decoder->get_parameters();
    encoder_params.insert(encoder_params.end(), decoder_params.begin(), decoder_params.end());
    return encoder_params;
}

std::vector<uint> Seq2SeqEncoderDecoder::predict(
    const std::vector<uint> &src_token_ids,
    uint max_len,
    std::vector<autograd::Node *> &enc_out_embs,
    std::vector<autograd::Node *> &dec_out_embs
) {
    std::vector<std::vector<uint>> v_src_token_ids;
    v_src_token_ids.push_back(src_token_ids);
    auto hiddens = encoder->forward(v_src_token_ids, {}, enc_out_embs);
    assert(hiddens.size() == 1);
    std::vector<std::vector<uint>> tgt_token_ids = {{bos_id}};

    for (uint step = 0; step < max_len; step++) {
        auto dec_outputs = decoder->forward(tgt_token_ids, hiddens, {}, dec_out_embs);
        assert(dec_outputs.size() == 1);
        assert(dec_outputs[0]->getShape().colCnt == step + 1);
        auto maxs = dec_outputs[0]->get_weight()->argMax();
        assert(maxs.size() == step + 1);
        auto predict_id = maxs[step];
        tgt_token_ids[0].push_back(predict_id);
        if (predict_id == eos_id) {
            break;
        }
    }
    return tgt_token_ids[0];
}