#ifndef SEQ2SEQ_DATALOADER_H
#define SEQ2SEQ_DATALOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include "dataloaders/vocab.h"

namespace seq2seq {

    class DataLoader {
    public:
        DataLoader(
            const std::string& _corpus_path,
            const std::string& _src_vocab_path,
            const std::string& _tgt_vocab_path,
            const std::string& _test_file
        );
        ~DataLoader();
        void get_token_ids(
            std::vector<std::vector<uint>>& src_token_ids,
            std::vector<std::vector<uint>>& tgt_token_ids
        );
        std::string get_src_token(uint token_id);
        std::string get_tgt_token(uint token_id);
        uint src_pad_id();
        uint tgt_pad_id();
        uint tgt_bos_id();
        uint tgt_eos_id();
        uint src_vocab_size();
        uint tgt_vocab_size();
        std::vector<uint> to_src_token_ids(const std::string& sentence);
        std::vector<std::string> get_test_sentences();
    private:
        std::string corpus_path;
        std::string src_vocab_path;
        std::string tgt_vocab_path;
        std::string test_file;
        ::Vocab src_vocab;
        ::Vocab tgt_vocab;
        std::vector<std::string> test_sentences;
    };
} // namespace seq2seq
#endif