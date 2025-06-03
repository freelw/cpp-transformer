#include "dataloader.h"

namespace seq2seq {

    Vocab::Vocab(const std::string& vocab_file) {
        id2token.push_back("<pad>");
        token2id["<pad>"] = 0;
        id2token.push_back("<eos>");
        token2id["<eos>"] = 1;
        id2token.push_back("<unk>");
        token2id["<unk>"] = 2;
        id2token.push_back("<bos>");
        token2id["<bos>"] = 3;
        std::ifstream ifs(vocab_file);
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            token2id[token] = id2token.size();
            id2token.push_back(token);
        }
    }

    Vocab::~Vocab() {}

    uint Vocab::get_token_id(const std::string& token) {
        auto it = token2id.find(token);
        if (it == token2id.end()) {
            return token2id["<unk>"];
        }
        return it->second;
    }

    std::string Vocab::get_token(uint token_id) {
        if (token_id >= id2token.size()) {
            return "<unk>";
        }
        return id2token[token_id];
    }

    uint Vocab::size() {
        return id2token.size();
    }

    DataLoader::DataLoader(
        const std::string& _corpus_path,
        const std::string& _src_vocab_path,
        const std::string& _tgt_vocab_path,
        const std::string& _test_file
    ) : corpus_path(_corpus_path),
        src_vocab_path(_src_vocab_path),
        tgt_vocab_path(_tgt_vocab_path),
        test_file(_test_file),
        src_vocab(_src_vocab_path),
        tgt_vocab(_tgt_vocab_path) {
        std::ifstream ifs(test_file);
        std::string line;
        while (std::getline(ifs, line)) {
            test_sentences.push_back(line);
        }
    }

    DataLoader::~DataLoader() {}
    void DataLoader::get_token_ids(
        std::vector<std::vector<uint>>& src_token_ids,
        std::vector<std::vector<uint>>& tgt_token_ids
    ) {
        std::ifstream ifs(corpus_path);
        std::string line;
        std::vector<uint> src_token_id;
        std::vector<uint> tgt_token_id;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string src_line, tgt_line;
            std::getline(iss, src_line, '\t');
            std::getline(iss, tgt_line);
            std::istringstream src_iss(src_line);
            std::istringstream tgt_iss(tgt_line);
            src_token_id.clear();
            tgt_token_id.clear();
            std::string token;
            while (src_iss >> token) {
                src_token_id.push_back(src_vocab.get_token_id(token));
            }
            src_token_id.push_back(src_vocab.get_token_id("<eos>"));
            while (tgt_iss >> token) {
                tgt_token_id.push_back(tgt_vocab.get_token_id(token));
            }
            tgt_token_id.push_back(tgt_vocab.get_token_id("<eos>"));
            src_token_ids.push_back(src_token_id);
            tgt_token_ids.push_back(tgt_token_id);
        }
    }

    std::string DataLoader::get_src_token(uint token_id) {
        return src_vocab.get_token(token_id);
    }

    std::string DataLoader::get_tgt_token(uint token_id) {
        return tgt_vocab.get_token(token_id);
    }

    uint DataLoader::src_pad_id() {
        return src_vocab.get_token_id("<pad>");
    }

    uint DataLoader::tgt_pad_id() {
        return tgt_vocab.get_token_id("<pad>");
    }

    uint DataLoader::tgt_bos_id() {
        return tgt_vocab.get_token_id("<bos>");
    }

    uint DataLoader::tgt_eos_id() {
        return tgt_vocab.get_token_id("<eos>");
    }

    uint DataLoader::src_vocab_size() {
        return src_vocab.size();
    }

    uint DataLoader::tgt_vocab_size() {
        return tgt_vocab.size();
    }

    std::vector<uint> DataLoader::to_src_token_ids(const std::string& sentence) {
        std::vector<uint> token_ids;
        std::istringstream iss(sentence);
        std::string token;
        while (iss >> token) {
            token_ids.push_back(src_vocab.get_token_id(token));
        }
        token_ids.push_back(src_vocab.get_token_id("<eos>"));
        return token_ids;
    }

    std::vector<std::string> DataLoader::get_test_sentences() {
        return test_sentences;
    }

} // namespace seq2seq