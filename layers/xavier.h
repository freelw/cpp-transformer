#ifndef XAVIEAR_H
#define XAVIEAR_H

#include "matrix/matrix.h"

namespace autograd {
    DATATYPE xavier_init_tanh(uint input_num, uint output_num);
    DATATYPE xavier_init_sigmoid(uint input_num, uint output_num);
    DATATYPE xavier_init_tanh(Matrix *m);
    DATATYPE xavier_init_sigmoid(Matrix *m);
    DATATYPE he_init_relu(Matrix *m);
} // namespace autograd


#endif