#include "checkpoint.h"
#include "backends/backend_ops.h"

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

void save_checkpoint(const std::string& prefix, int epoch, const std::vector<Parameter*>& parameters) {
    std::ostringstream oss;
    oss << prefix << "_" << epoch << ".bin";
    std::string checkpoint_name = oss.str();
    std::string path = "./checkpoints/" + checkpoint_name;
    std::ofstream out(path, std::ios::out | std::ios::binary);
    int num_params = parameters.size();
    out.write((char*)&num_params, sizeof(num_params));
    for (auto p : parameters) {
        std::string serialized = p->serialize();
        int size = serialized.size();
        out.write((char*)&size, sizeof(size));
        out.write(serialized.c_str(), serialized.size());
    }
    out.close();
    std::cout << "checkpoint saved : " << path << std::endl;
}

void loadfrom_checkpoint(const std::string& filename, std::vector<Parameter*>& parameters) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    // check file exists
    if (!in) {
        std::cerr << "file not found : " << filename << std::endl;
        exit(1);
    }
    int num_params = 0;
    in.read((char*)&num_params, sizeof(num_params));
    assert(num_params == parameters.size());
    for (int i = 0; i < num_params; i++) {
        int size;
        in.read((char*)&size, sizeof(size));
        assert(size == parameters[i]->get_serialized_size());
        char* buffer = static_cast<char*>(::malloc(size));
        in.read(buffer, size);
        parameters[i]->deserialize(buffer);
        ::free(buffer);
    }
}

void diff_tensor_buffer(Tensor* tensor, char* buffer) {
    int size = tensor->size();
    int length = tensor->length();
    char* tensor_buffer = static_cast<char*>(::malloc(size));
    g_backend_ops->cp_from_device(tensor_buffer, tensor, size);
    float* tensor_buffer_f = reinterpret_cast<float*>(tensor_buffer);
    float* buffer_f = reinterpret_cast<float*>(buffer);
    const float eps = 1e-4f;

    for (int i = 0; i < length; ++i) {
        if (std::abs(tensor_buffer_f[i] - buffer_f[i]) > eps) {
            std::cerr << "diff tensor failed at index " << i
                << ", expected: " << tensor_buffer_f[i]
                << ", got: " << buffer_f[i] << std::endl;
            std::cerr << "tensor meta : " << tensor->get_meta_info() << std::endl;
            break;
        }
    }
    ::free(tensor_buffer);
}

void diff_para(Parameter* p, ParameterInfo* info) {
    Tensor* w = p->get_w();
    Tensor* m = p->get_m();
    Tensor* v = p->get_v();
    if (w->size() != info->weight_size) {
        std::cerr << "weight size mismatch: expected " << info->weight_size
            << ", got " << w->size() << std::endl;
        std::cerr << "parameter meta : " << p->get_w()->get_meta_info() << std::endl;
        abort();
    }
    if (m->size() != info->m_size) {
        std::cerr << "m size mismatch: expected " << info->m_size
            << ", got " << m->size() << std::endl;
        std::cerr << "parameter meta : " << p->get_m()->get_meta_info() << std::endl;
        abort();
    }
    if (v->size() != info->v_size) {
        std::cerr << "v size mismatch: expected " << info->v_size
            << ", got " << v->size() << std::endl;
        std::cerr << "parameter meta : " << p->get_v()->get_meta_info() << std::endl;
        abort();
    }
    if (p->get_t() != info->t) {
        std::cerr << "t mismatch: expected " << info->t
            << ", got " << p->get_t() << std::endl;
        std::cerr << "parameter meta : " << p->get_w()->get_meta_info() << std::endl;
        abort();
    }

    diff_tensor_buffer(w, info->weight_start);
    diff_tensor_buffer(m, info->m_start);
    diff_tensor_buffer(v, info->v_start);
    std::cout << "diff parameter success : " << w->get_meta_info() << std::endl;
    std::cout << "diff parameter success : " << m->get_meta_info() << std::endl;
    std::cout << "diff parameter success : " << v->get_meta_info() << std::endl;
}

void difffrom_checkpoint(const std::string& filename, std::vector<Parameter*>& parameters) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    // check file exists
    if (!in) {
        std::cerr << "file not found : " << filename << std::endl;
        exit(1);
    }
    int num_params = 0;
    in.read((char*)&num_params, sizeof(num_params));
    assert(num_params == parameters.size());
    bool succ = true;
    for (int i = 0; i < num_params; i++) {
        int size;
        in.read((char*)&size, sizeof(size));
        assert(size == parameters[i]->get_serialized_size());
        char* buffer = static_cast<char*>(::malloc(size));
        in.read(buffer, size);
        ParameterInfo info;
        parameters[i]->deserialize_info(buffer, &info);
        diff_para(parameters[i], &info);
        ::free(buffer);
    }
}