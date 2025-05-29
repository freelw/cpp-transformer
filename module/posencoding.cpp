#include "posencoding.h"
#include "graph/actions.h"

PosEncoding::PosEncoding(int _max_len, int _num_hidden, float p)
    :max_len(_max_len), num_hidden(_num_hidden) {
    pos_enc = allocTensor({max_len, num_hidden}, "pos_enc"); // do not calloc
    gCreateAction(
        new PosEncodingAction(
            pos_enc
        )
    );
    dropout = new Dropout(p);
}

PosEncoding::~PosEncoding() {
    assert(dropout != nullptr);
    delete dropout;
}

graph::Node *PosEncoding::forward(graph::Node *input) {
    assert(input->get_tensor()->get_dim() == 3);
    auto origin_shape = input->get_tensor()->get_shape();
    auto shape = input->get_tensor()->get_shape();
    assert(shape[2] == num_hidden);
    if (!input->get_tensor()->is_contiguous()) {
        input = input->reshape(shape);
    }
    auto cp_size = shape[1] * num_hidden * sizeof(float);
    Tensor *pe = callocTensor(
        {shape[0], shape[1], num_hidden},
        "pos_enc"
    );
    for (int i = 0; i < shape[0]; ++ i) {
        auto offset = i * cp_size;
        gCreateAction(
            new MemCpAction(
                pe,
                pos_enc,
                offset,
                0,
                cp_size
            )
        );
    }
    auto npe = graph::allocNode(pe);
    std::vector<int> add_shape = {-1, num_hidden};
    assert(input->get_tensor()->is_contiguous());
    assert(npe->get_tensor()->is_contiguous());
    return dropout->forward(input->add(npe))->reshape(origin_shape);
}
