#ifndef VOCAB_H
#define VOCAB_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

class Vocab {
public:
    Vocab(const std::string& vocab_file);
    ~Vocab();
    uint get_token_id(const std::string& token);
    std::string get_token(uint token_id);
    uint size();
private:
    std::map<std::string, uint> token2id;
    std::vector<std::string> id2token;
};
#endif