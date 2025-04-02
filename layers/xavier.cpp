#include "xavier.h"

namespace autograd {
    DATATYPE xavier_init_tanh(uint input_num, uint output_num) {
        return sqrt(2.0 / (input_num + output_num)) * 4;
    }

    DATATYPE xavier_init_sigmoid(uint input_num, uint output_num) {
        return sqrt(2.0 / (input_num + output_num));
    }

    DATATYPE xavier_init_tanh(Matrix *m) {
        return xavier_init_tanh(m->getShape().colCnt, m->getShape().rowCnt);
    }

    DATATYPE xavier_init_sigmoid(Matrix *m) {
        return xavier_init_sigmoid(m->getShape().colCnt, m->getShape().rowCnt);
    }

    DATATYPE he_init_relu(uint input_num, uint output_num) {
        return sqrt(2.0 / input_num);
    }

    DATATYPE he_init_relu(Matrix *m) {
        return he_init_relu(m->getShape().colCnt, m->getShape().rowCnt);
    }
} // namespace autograd