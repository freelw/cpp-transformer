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
        const std::string& _test_file
    );
    ~LMDataLoader();
    void get_token_ids(
        std::vector<uint>& tgt_token_ids
    );
    std::string get_tgt_token(uint token_id);
    uint tgt_pad_id();
    uint tgt_bos_id();
    uint tgt_eos_id();
    uint tgt_vocab_size();
    std::vector<std::string> get_test_sentences();
private:
    std::string corpus_path;
    std::string src_vocab_path;
    std::string tgt_vocab_path;
    std::string test_file;
    Vocab src_vocab;
    Vocab tgt_vocab;
    std::vector<std::string> test_sentences;
};
#endif