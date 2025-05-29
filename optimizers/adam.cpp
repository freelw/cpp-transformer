#include "adam.h"
#include "backends/backend_ops.h"

void Adam::step() {
    for (auto param : parameters) {
        assert(param->is_require_grad());
        assert(param->get_grad() != nullptr);
        gCreateAction(
            new AdamStepAction(param, lr, beta1, beta2, epsilon)
        );
    }
}

void Adam::clip_grad(float grad_clip_val) {
    std::vector<Tensor *> grads;
    grads.reserve(parameters.size());
    for (auto param : parameters) {
        assert(param->is_require_grad());
        assert(param->get_grad() != nullptr);
        grads.push_back(param->get_grad());
    }
    Tensor *norm = callocTensor({1}, "clip_grad_norm");
    gCreateAction(
        new CalcAllGradNormAction(grads, norm)
    );
    for (auto grad: grads) {
        gCreateAction(
            new ClipGradAction(grad, norm, grad_clip_val)
        );
    }
}