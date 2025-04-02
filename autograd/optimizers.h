#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "autograd/node.h"
#include "autograd/parameters.h"

namespace autograd {
    class Adam {
        public:
            Adam(std::vector<Parameters*> _parameters, DATATYPE _lr,
                DATATYPE _beta1 = 0.9, DATATYPE _beta2 = 0.999, DATATYPE _epsilon = 1e-8)
                : parameters(_parameters), lr(_lr),
                    beta1(_beta1), beta2(_beta2), epsilon(_epsilon)
                {}
            void step();
            void zero_grad();
            bool clip_grad(DATATYPE grad_clip_val);
        private:
            std::vector<Parameters*> parameters;
            DATATYPE lr; 
            DATATYPE beta1;
            DATATYPE beta2;
            DATATYPE epsilon;
    };
} // namespace autograd
#endif