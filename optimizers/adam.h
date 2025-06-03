#ifndef ADAM_H
#define ADAM_H

#include "parameter.h"

class Adam {
public:
    Adam(const std::vector<Parameter*>& _parameters, float _lr,
        float _beta1 = 0.9, float _beta2 = 0.999, float _epsilon = 1e-20)
        : parameters(_parameters), lr(_lr),
        beta1(_beta1), beta2(_beta2), epsilon(_epsilon) {
    }
    void step();
    void clip_grad(float grad_clip_val);

private:
    std::vector<Parameter*> parameters;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
};
#endif