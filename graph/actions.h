#ifndef ACTIONS_H
#define ACTIONS_H

#include "tensor/tensor.h"
#include <ostream>
#include <string>

class Parameter;

class Action {
public:
    Action(Tensor* _lhs, const Tensor* _rhs, Tensor* _res)
        : lhs(_lhs), rhs(_rhs), res(_res), exec_times(0) {
    }
    virtual ~Action() = default;
    virtual void execute() = 0;
    virtual std::string to_string() const {
        return "Action not implemented";
    }
    virtual bool is_do_once() const {
        return false;
    }
    virtual bool is_backward_boundary() const {
        return false;
    }
    virtual bool is_zero_c_tensors() const {
        return false;
    }
    virtual bool is_zero_grad() const {
        return false;
    }
    virtual bool is_init_weight() const {
        return false;
    }
    bool executed_once() const;
    void increase_exec_times();
    int get_exec_times() const;
    friend std::ostream& operator<<(std::ostream& output, const Action&);
protected:
    Tensor* lhs;
    const Tensor* rhs;
    Tensor* res;
    int exec_times;
};

class AddAction : public Action {
public:
    AddAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res);
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* lhs_shape;
    Tensor* lhs_strides;
    Tensor* rhs_strides;
    Tensor* res_strides;
};

// AddEqAction 永远不会出现在forward中
class AddEqAction : public Action {
public:
    AddEqAction(Tensor* _lhs, const Tensor* _rhs);
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* lhs_shape;
    Tensor* lhs_strides;
    Tensor* rhs_strides;
};

class ExpandAddAction : public Action {
public:
    ExpandAddAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res)
        : Action(_lhs, _rhs, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class ExpandMulAction : public Action {
public:
    ExpandMulAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res)
        : Action(_lhs, _rhs, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class AtAction : public Action {
public:
    AtAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res)
        : Action(_lhs, _rhs, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class MulAction : public Action {
public:
    MulAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res);
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* lhs_shape;
    Tensor* lhs_strides;
    Tensor* rhs_strides;
    Tensor* res_strides;
};

class SumAction : public Action {
public:
    SumAction(Tensor* _lhs, Tensor* _res, int _dim)
        : Action(_lhs, nullptr, _res), dim(_dim) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    int dim;
};

class ReluAction : public Action {
public:
    ReluAction(Tensor* _lhs, Tensor* _res)
        : Action(_lhs, nullptr, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class ReluPrimeAction : public Action {
public:
    ReluPrimeAction(Tensor* _lhs, Tensor* _res)
        : Action(_lhs, nullptr, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class CrossEntropyAction : public Action {
public:
    CrossEntropyAction(Tensor* _lhs, const Tensor* labels, Tensor* _maxs, Tensor* _sums, Tensor* _res)
        : Action(_lhs, labels, _res), maxs(_maxs), sums(_sums) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* maxs;
    Tensor* sums;
};

class CrossEntropyBackwardAction : public Action {
public:
    CrossEntropyBackwardAction(Tensor* _lhs, const Tensor* labels, Tensor* _maxs, Tensor* _sums, Tensor* _res)
        : Action(_lhs, labels, _res), maxs(_maxs), sums(_sums) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* maxs;
    Tensor* sums;
};

class CalcAllGradNormAction : public Action {
public:
    CalcAllGradNormAction(const std::vector<Tensor*>& _grads, Tensor* _norm)
        : Action(nullptr, nullptr, _norm), grads(_grads) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    std::vector<Tensor*> grads;
};

class ClipGradAction : public Action {
public:
    ClipGradAction(Tensor* _grad, Tensor* _norm, float _grad_clip_val)
        : Action(_grad, _norm, nullptr), grad_clip_val(_grad_clip_val) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    float grad_clip_val;
};

class AdamStepAction : public Action {
public:
    AdamStepAction(Parameter* _param, float _lr, float _beta1, float _beta2, float _epsilon)
        : Action(nullptr, nullptr, nullptr), param(_param), lr(_lr), beta1(_beta1), beta2(_beta2), epsilon(_epsilon) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    Parameter* param;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
};

class ZeroGradAction : public Action {
public:
    ZeroGradAction()
        : Action(nullptr, nullptr, nullptr) {
    }
    void execute() override;
    std::string to_string() const override;
    bool is_zero_grad() const override {
        return true;
    }
};

class ZeroCTensorsAction : public Action {
public:
    ZeroCTensorsAction()
        : Action(nullptr, nullptr, nullptr) {
    }
    bool is_zero_c_tensors() const override {
        return true;
    }
    void execute() override;
    std::string to_string() const override;
};

class PrintNoZeroTensorNamesAction : public Action {
public:
    PrintNoZeroTensorNamesAction()
        : Action(nullptr, nullptr, nullptr) {
    }
    void execute() override;
    std::string to_string() const override;
};

class FillWeightAction : public Action {
public:
    FillWeightAction(Tensor* _lhs, const std::string& _init_type, float _sigma, float _mean)
        : Action(_lhs, nullptr, nullptr), init_type(_init_type), sigma(_sigma), mean(_mean) {
    }
    void execute() override;
    std::string to_string() const override;
protected:
    std::string init_type;
    float sigma;
    float mean;
};

class InitWeightAction : public FillWeightAction {
public:
    InitWeightAction(Tensor* _lhs, const std::string& _init_type, float _sigma, float _mean)
        : FillWeightAction(_lhs, _init_type, _sigma, _mean) {
    }
    bool is_do_once() const override {
        return true;
    }
    bool is_init_weight() const override {
        return true;
    }
    std::string to_string() const override;
};

class BoundaryAction : public Action {
public:
    BoundaryAction()
        : Action(nullptr, nullptr, nullptr) {
    }
    void execute() override;
    bool is_backward_boundary() const override;
    std::string to_string() const override;
};

class AssignShapeAndStridesAction : public Action {
public:
    AssignShapeAndStridesAction(
        Tensor* tensor_shape,
        Tensor* tensor_strides,
        const std::vector<int>& shape,
        const std::vector<int>& strides
    );
    virtual ~AssignShapeAndStridesAction();
    void execute() override;
    std::string to_string() const override;
private:
    int32_t* shape_data;
    int32_t* strides_data;
};

class ReshapeDeepCpAction : public Action {
public:
    ReshapeDeepCpAction(
        Tensor* _lhs, const Tensor* _rhs,
        const Tensor* _shape, const Tensor* _strides)
        : Action(_lhs, _rhs, nullptr),
        shape(_shape), strides(_strides) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    const Tensor* shape;
    const Tensor* strides;
};

class RepeatInterleaveAction : public Action {
public:
    RepeatInterleaveAction(Tensor* _lhs, Tensor* _res, int _n)
        : Action(_lhs, nullptr, _res), n(_n) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    int n;
};

class SequenceMaskAction : public Action {
public:
    SequenceMaskAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res, float _value)
        : Action(_lhs, _rhs, _res), value(_value) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    float value;
};

class SoftmaxAction : public Action {
public:
    SoftmaxAction(Tensor* _lhs, Tensor* _res)
        : Action(_lhs, nullptr, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class SoftmaxBackwardAction : public Action {
public:
    SoftmaxBackwardAction(Tensor* _lhs, Tensor* _softmax_res, Tensor* grad)
        : Action(_lhs, _softmax_res, grad) {
    }
    void execute() override;
    std::string to_string() const override;
};

class DivAction : public Action {
public:
    DivAction(Tensor* _lhs, Tensor* _res, float _value)
        : Action(_lhs, nullptr, _res), value(_value) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    float value;
};

class LazyDivAction : public Action {
public:
    LazyDivAction(Tensor* _lhs, Tensor* _res, Tensor* _value)
        : Action(_lhs, nullptr, _res), value(_value) {
        assert(value->get_dim() == 1);
        assert(value->get_shape()[0] == 1);
    }
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* value;
};

class DropoutMaskAction : public Action {
public:
    DropoutMaskAction(Tensor* mask, float _p);
    void execute() override;
    std::string to_string() const override;
private:
    float p;
    Tensor* shape;
    Tensor* strides;
};

class EmbeddingAction : public Action {
public:
    EmbeddingAction(Tensor* _lhs, Tensor* indices, Tensor* _res)
        : Action(_lhs, indices, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class EmbeddingBackwardAction : public Action {
public:
    EmbeddingBackwardAction(Tensor* _lhs, const Tensor* indices, Tensor* _res)
        : Action(_lhs, indices, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class PosEncodingAction : public Action {
public:
    PosEncodingAction(Tensor* res)
        : Action(nullptr, nullptr, res) {
    }
    void execute() override;
    std::string to_string() const override;
    bool is_do_once() const override {
        return true;
    }
};

class AvgAction : public Action {
public:
    AvgAction(Tensor* _lhs, Tensor* _res)
        : Action(_lhs, nullptr, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class VarAction : public Action {
public:
    VarAction(Tensor* _lhs, Tensor* avg, Tensor* _res)
        : Action(_lhs, avg, _res) {
    }
    void execute() override;
    std::string to_string() const override;
};

class NormAction : public Action {
public:
    NormAction(Tensor* _src, Tensor* avg, Tensor* var, Tensor* _res)
        : Action(avg, var, _res), src(_src) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* src;
};

class NormBackwardAction : public Action {
public:
    NormBackwardAction(Tensor* _grad, Tensor* norm_res, Tensor* _var_tensor, Tensor* _res)
        : Action(_grad, norm_res, _res), var_tensor(_var_tensor) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    Tensor* var_tensor;
};

class DbgPrintAction : public Action {
public:
    DbgPrintAction(Tensor* _lhs, const std::string& _msg, const std::string& _expected_name = "")
        : Action(_lhs, nullptr, nullptr), msg(_msg), expected_name(_expected_name) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    std::string msg;
    std::string expected_name;
};

class MemCpAction : public Action {
public:
    MemCpAction(Tensor* _lhs, const Tensor* _rhs, int _offset_l, int _offset_r, int _size)
        : Action(_lhs, _rhs, nullptr), offset_l(_offset_l), offset_r(_offset_r), size(_size) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    int offset_l;
    int offset_r;
    int size;
};

class MulSVAction : public Action {
public:
    MulSVAction(Tensor* _lhs, Tensor* _res, float _value)
        : Action(_lhs, nullptr, _res), value(_value) {
    }
    void execute() override;
    std::string to_string() const override;
private:
    float value;
};

class ClearAction : public Action {
public:
    ClearAction(Tensor* _lhs)
        : Action(_lhs, nullptr, nullptr) {
    }
    void execute() override;
    std::string to_string() const override;
};

std::vector<Action*> getOnceActions();
void gCreateAction(Action* action);
void gDoActions();
void gDoOnceActions();
void gDoForwardActions(bool training = false);
void gDoBackwardActions();
void printAllActions();
void freeAllActions();

void disableInitWeightAction();
// for test
void disableOnceAction();


#endif