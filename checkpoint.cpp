
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "checkpoint.h"


std::string generateDateTimeSuffix() {    
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    struct std::tm* localTime = std::localtime(&currentTime);
    std::ostringstream oss;
    oss << std::put_time(localTime, "_%Y%m%d_%H%M%S");
    return oss.str();
}

void save_checkpoint(const std::string & prefix, int epoch, Seq2SeqEncoderDecoder &lm) {
    std::ostringstream oss;
    oss << prefix << "_" << epoch << ".bin";
    std::string checkpoint_name = oss.str();
    std::string path = "./checkpoints/" + checkpoint_name;
    auto parameters = lm.get_parameters();
    std::ofstream out(path, std::ios::out | std::ios::binary);
    int num_params = parameters.size();
    out.write((char *)&num_params, sizeof(num_params));
    for (auto p : parameters) {
        std::string serialized = p->serialize();
        int size = serialized.size();
        out.write((char *)&size, sizeof(size));
        out.write(serialized.c_str(), serialized.size());
    }
    out.close();
    std::cout << "checkpoint saved : " << path << std::endl;
}

void loadfrom_checkpoint(Seq2SeqEncoderDecoder &lm, const std::string &filename) {
    std::ifstream in(filename
        , std::ios::in | std::ios::binary);
    // check file exsit
    if (!in) {
        std::cerr << "file not found : " << filename << std::endl;
        exit(1);
    }
    int num_params = 0;    
    in.read((char *)&num_params, sizeof(num_params));
    for (int i = 0; i < num_params; i++) {
        int size;
        in.read((char *)&size, sizeof(size));
        char *buffer = new char[size];
        in.read(buffer, size);
        lm.get_parameters()[i]->deserialize(buffer);
        delete [] buffer;
    }
}