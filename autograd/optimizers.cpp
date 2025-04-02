#include "autograd/optimizers.h"
#include <iostream> 

namespace autograd {
    void Adam::step() {
        for (auto p : parameters) {
            if (!p->require_grad()) {
                continue;
            }
            p->inc_t();
            auto t = p->get_t();
            Matrix *weight = p->get_weight();
            Matrix *grad = p->get_grad();
            Matrix *mm = p->get_m();
            Matrix *mv = p->get_v();
            Shape shape = weight->getShape();
            grad->checkShape(shape);
            mm->checkShape(shape);
            mv->checkShape(shape);
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    auto &value = (*weight)[i][j];
                    auto &m = (*mm)[i][j];
                    auto &v = (*mv)[i][j];
                    auto &gradient = (*grad)[i][j];
                    m = beta1 * m + (1 - beta1) * gradient;
                    v = beta2 * v + (1 - beta2) * std::pow(gradient, 2);
                    DATATYPE m_hat = m / (1 - std::pow(beta1, t));
                    DATATYPE v_hat = v / (1 - std::pow(beta2, t));
                    value -=  lr * (m_hat / (std::sqrt(v_hat) + epsilon));
                }
            }
        }
    }

    void Adam::zero_grad() {
        for (auto p : parameters) {
            if (!p->require_grad()) {
                continue;
            }
            p->zero_grad();
        }
    }

    bool Adam::clip_grad(DATATYPE grad_clip_val) {
        DATATYPE norm = 0;
        
        for (auto param : parameters) {
            if (!param->require_grad()) {
                continue;
            }
            auto grad = param->get_grad();
            Shape shape = grad->getShape();
            for (uint i = 0; i < shape.rowCnt; ++ i) {
                for (uint j = 0; j < shape.colCnt; ++ j) {
                    norm += std::pow((*grad)[i][j], 2);
                }
            }
        }
        norm = sqrt(norm);
        if (norm > grad_clip_val) {
            for (auto param : parameters) {
                if (!param->require_grad()) {
                    continue;
                }
                auto grad = param->get_grad();
                *grad *= grad_clip_val / norm;
            }
        }
        return norm > grad_clip_val;
    }

} // namespace autograd