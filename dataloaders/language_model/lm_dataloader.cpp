#include "lm_dataloader.h"

LMDataLoader::LMDataLoader(
    const std::string& _corpus_path,
    const std::string& _tgt_vocab_path,
    const std::string& _test_file
)
    :corpus_path(_corpus_path),
    tgt_vocab_path(_tgt_vocab_path),
    test_file(_test_file),
    tgt_vocab(_tgt_vocab_path) {
    std::ifstream ifs(test_file);
    std::string line;
    while (std::getline(ifs, line)) {
        test_sentences.push_back(line);
    }
}

void LMDataLoader::get_token_ids(std::vector<uint>& tgt_token_ids) {
    std::ifstream ifs(corpus_path);
    std::string token;
    while (ifs >> token) {
        tgt_token_ids.push_back(tgt_vocab.get_token_id(token));
    }
}

std::string LMDataLoader::get_tgt_token(uint token_id) {
    return tgt_vocab.get_token(token_id);
}

uint LMDataLoader::tgt_pad_id() {
    return tgt_vocab.get_token_id("<pad>");
}

uint LMDataLoader::tgt_vocab_size() {
    return tgt_vocab.size();
}

std::vector<std::string> LMDataLoader::get_test_sentences() {
    return test_sentences;
}