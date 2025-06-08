#include "lm_dataloader.h"
#include <algorithm>

LMDataLoader::LMDataLoader(
    const std::string& _corpus_path,
    const std::string& _tgt_vocab_path,
    const std::string& _test_file,
    const int _num_steps
)
    :corpus_path(_corpus_path),
    tgt_vocab_path(_tgt_vocab_path),
    test_file(_test_file),
    tgt_vocab(_tgt_vocab_path),
    num_steps(_num_steps) {
    std::ifstream ifs(test_file);
    std::string line;
    while (std::getline(ifs, line)) {
        test_sentences.push_back(line);
    }
}

void LMDataLoader::get_token_ids(
    std::vector<std::vector<uint>>& v_src_token_ids,
    std::vector<std::vector<uint>>& v_tgt_token_ids
) {
    std::ifstream ifs(corpus_path);
    std::string token;
    std::vector<uint> token_ids;
    while (ifs >> token) {
        token_ids.push_back(tgt_vocab.get_token_id(token));
    }

    int token_ids_size = std::min((int)token_ids.size(), 256);

    for (size_t i = 0; i < token_ids_size; ++i) {
        std::vector<uint> src_step_tokens;
        std::vector<uint> tgt_step_tokens;
        for (size_t j = 0; j < num_steps && (i + j) < token_ids_size - 1; ++j) {
            src_step_tokens.push_back(token_ids[i + j]);
            tgt_step_tokens.push_back(token_ids[i + j + 1]);
        }
        v_src_token_ids.push_back(src_step_tokens);
        v_tgt_token_ids.push_back(tgt_step_tokens);
    }
}

std::string LMDataLoader::get_tgt_token(uint token_id) {
    return tgt_vocab.get_token(token_id);
}

uint LMDataLoader::get_pad_id() {
    return tgt_vocab.get_token_id("<pad>");
}

uint LMDataLoader::tgt_vocab_size() {
    return tgt_vocab.size();
}

std::vector<std::string> LMDataLoader::get_test_sentences() {
    return test_sentences;
}