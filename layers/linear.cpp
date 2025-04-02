#include "linear.h"
#include "xavier.h"
#include "macro.h"

namespace autograd {
    Linear::Linear(uint input_num, uint output_num, ACTIVATION act, bool _bias) 
        : bias(_bias) {
        mW = new Matrix(Shape(output_num, input_num));
        #ifdef DEBUG_GRAD
            #pragma message("DEBUG_GRAD")
            mW->fill(1);
            (*mW)[0][0] = 0.1;
            // assert(input_num == output_num); // for debug
            // mW->fill(0);
            // for (uint i = 0; i < output_num; i++) {
            //     (*mW)[i][i] = 1;
            // }
        #else
            switch (act) {
                case RELU:
                    init_weight(mW, he_init_relu(mW));
                    break;
                case SIGMOID:
                    init_weight(mW, xavier_init_sigmoid(mW));
                    break;
                case TANH:
                    init_weight(mW, xavier_init_tanh(mW));
                    break;
                case NONE:
                    init_weight(mW, xavier_init_sigmoid(mW));
                    break;
                default:
                    assert(false);
            }
        #endif
        W = new Node(mW, true);
        W->require_grad();
        PW = new Parameters(W);

        if (bias) {
            mb = new Matrix(Shape(output_num, 1));
            // mb->zero();
            init_weight(mb, 0.01, 0.01); // this is very important, break the symmetry
            b = new Node(mb, true);
            b->require_grad();
            Pb = new Parameters(b);
        }
    }

    Linear::~Linear() {
        delete mW;
        delete W;
        delete PW;
        if (bias) {
            delete mb;
            delete b;
            delete Pb;
        }
    }

    Node *Linear::forward(Node *input) {
        auto node = W->at(input);
        return bias ? node->expand_add(b) : node;
    }

    std::vector<Node *> Linear::forward(const std::vector<Node *> &input) {
        std::vector<Node *> res;
        for (auto node : input) {
            res.push_back(forward(node));
        }
        return res;
    }

    std::vector<Parameters *> Linear::get_parameters() {
        std::vector<Parameters *> res;
        res.push_back(PW);
        if (bias) {
            res.push_back(Pb);
        }
        return res;
    }

    LazyLinear::LazyLinear(uint _output_num, ACTIVATION _act, bool _bias) 
        : output_num(_output_num), bias(_bias), linear(nullptr), act(_act) {
    }

    LazyLinear::~LazyLinear() {
        if (linear != nullptr) {
            delete linear;
        }
    }

    Node *LazyLinear::forward(Node *input) {
        if (linear == nullptr) {
            linear = new Linear(input->get_weight()->getShape().rowCnt, output_num, act, bias);
        }
        return linear->forward(input);
    }

    std::vector<Node *> LazyLinear::forward(const std::vector<Node *> &input) {
        std::vector<Node *> res;
        for (auto node : input) {
            res.push_back(forward(node));
        }
        return res;
    }

    std::vector<Parameters *> LazyLinear::get_parameters() {
        assert(linear != nullptr);
        return linear->get_parameters();
    }
    
} // namespace autograd
