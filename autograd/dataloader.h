#ifndef AUTOGRAD_DATALOADER_H
#define AUTOGRAD_DATALOADER_H


#include <fstream>
#include <assert.h>
#include "matrix/matrix.h"
#include <vector>
#include "lmcommon/common.h"

#define INPUT_NUM 28

namespace autograd {

    class DataLoader {
        public:
            DataLoader(const std::string &filename) {
                std::ifstream ifs(filename);
                std::string content((std::istreambuf_iterator<char>(ifs)),
                                    (std::istreambuf_iterator<char>()));
                this->content = content;
            }
            ~DataLoader() {}
        public:
            std::string content;
    };
} // namespace autograd
#endif