#include "vocab.h"

Vocab::Vocab(const std::string& vocab_file) {
    std::cout << "Loading vocabulary from: " << vocab_file << std::endl;
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
