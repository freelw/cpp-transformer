#ifndef POSENCODING_H
#define POSENCODING_H

#include "graph/node.h"
#include "module/dropout.h"

class PosEncoding {
    public:
        PosEncoding(int _max_len, int _num_hidden, float p);
        ~PosEncoding();
        graph::Node *forward(graph::Node *input);
    private:
        int max_len;
        int num_hidden;
        Tensor *pos_enc;
        Dropout *dropout;
};

#endif