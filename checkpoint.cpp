#include "checkpoint.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

std::string generateDateTimeSuffix() {    
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    struct std::tm* localTime = std::localtime(&currentTime);
    std::ostringstream oss;
    oss << std::put_time(localTime, "_%Y%m%d_%H%M%S");
    return oss.str();
}

void save_checkpoint(const std::string & prefix, int epoch, const std::vector<Parameter*> &parameters) {
    std::ostringstream oss;
    oss << prefix << "_" << epoch << ".bin";
    std::string checkpoint_name = oss.str();
    std::string path = "./checkpoints/" + checkpoint_name;
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

void loadfrom_checkpoint(const std::string &filename, std::vector<Parameter*> &parameters) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    // check file exists
    if (!in) {
        std::cerr << "file not found : " << filename << std::endl;
        exit(1);
    }
    int num_params = 0;    
    in.read((char *)&num_params, sizeof(num_params));
    assert(num_params == parameters.size());
    for (int i = 0; i < num_params; i++) {
        int size;
        in.read((char *)&size, sizeof(size));
        assert(size == parameters[i]->get_serialized_size());
        char *buffer = static_cast<char*>(::malloc(size));
        in.read(buffer, size);
        parameters[i]->deserialize(buffer);
        ::free(buffer);
    }
}