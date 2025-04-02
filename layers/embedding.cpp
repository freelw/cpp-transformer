#include "embedding.h"
#include "macro.h"

namespace autograd {

    Embedding::Embedding(uint _vocab_size, uint _hidden_num) : vocab_size(_vocab_size), hidden_num(_hidden_num) {
        for (uint i = 0; i < vocab_size; i++) {
            Matrix *m = new Matrix(Shape(hidden_num, 1));
            #ifdef DEBUG_GRAD
                #pragma message("Embedding DEBUG_GRAD")
                m->fill(1);
                (*m)[0][0] = 0.1*i;
            #else
                init_weight_uniform(m, sqrt(1.0/hidden_num));
            #endif
            mW.push_back(m);
            Node *n = new Node(m, true);
            n->require_grad();
            W.push_back(n);
            PW.push_back(new Parameters(n));
        }
    }

    Embedding::~Embedding() {
        for (auto m : mW) {
            delete m;
        }
        for (auto n : W) {
            delete n;
        }
        for (auto p : PW) {
            delete p;
        }
    }

    std::vector<Node *> Embedding::forward(const std::vector<std::vector<uint>> &inputs) {
        std::vector<Node *> res;
        for (auto &input : inputs) {
            std::vector<Node *> tmp;
            for (auto i : input) {
                tmp.push_back(W[i]);
            }
            res.push_back(cat(tmp));
        }
        return res;
    }

    std::vector<Parameters *> Embedding::get_parameters() {
        std::vector<Parameters *> res;
        for (auto p : PW) {
            res.push_back(p);
        }
        return res;
    }

} // namespace autograd