#ifndef LM_DADALOADER_H
#define LM_DADALOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include "dataloaders/vocab.h"

class LMDataLoader {
public:
    LMDataLoader(
        const std::string& _corpus_path,
        const std::string& _tgt_vocab_path,
        const std::string& _test_file,
        const int _num_steps
    );
    ~LMDataLoader() = default;
    void get_token_ids(
        std::vector<std::vector<uint>>& v_src_token_ids,
        std::vector<std::vector<uint>>& v_tgt_token_ids
    );
    std::string get_tgt_token(uint token_id);
    uint get_tgt_token_id(const std::string& token);
    uint get_pad_id();
    uint tgt_vocab_size();
    std::vector<std::string> get_test_sentences();
private:
    std::string corpus_path;
    std::string tgt_vocab_path;
    std::string test_file;
    Vocab tgt_vocab;
    std::vector<std::string> test_sentences;
    int num_steps;
};
#endif