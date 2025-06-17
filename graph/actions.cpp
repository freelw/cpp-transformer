#include "actions.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <fstream>
#include "backends/backend_ops.h"
#include "optimizers/parameter.h"

extern bool g_training;
int g_action_id_counter = 0;

bool Action::executed_once() const {
    return exec_times > 0;
}

void Action::increase_exec_times() {
    exec_times++;
}

int Action::get_exec_times() const {
    return exec_times;
}

std::string Action::get_dot_string() {
    std::ostringstream oss;
    if (lhs) {
        oss << "Tensor_" << lhs->get_id() << " -> " << "Action_" << action_id << ";" << std::endl;
    }
    if (rhs) {
        oss << "Tensor_" << rhs->get_id() << " -> " << "Action_" << action_id << ";" << std::endl;
    }
    if (res) {
        oss << "Action_" << action_id << " -> " << "Tensor_" << res->get_id() << ";" << std::endl;
    }
    return oss.str();
}

std::ostream& operator<<(std::ostream& output, const Action& a) {
    output << a.to_string();
    return output;
}

AddAction::AddAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res)
    : Action(_lhs, _rhs, _res) {

    assert(_lhs->get_dim() == _rhs->get_dim());
    assert(_lhs->get_shape() == _rhs->get_shape());
    auto dim = _lhs->get_dim();

    lhs_shape = callocTensor(
        { dim },
        _lhs->get_name() + "_shape",
        INT32
    );
    lhs_strides = callocTensor(
        { dim },
        _lhs->get_name() + "_strides",
        INT32
    );
    rhs_strides = callocTensor(
        { dim },
        _rhs->get_name() + "_strides",
        INT32
    );
    res_strides = callocTensor(
        { dim },
        _res->get_name() + "_strides",
        INT32
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            lhs_shape,
            lhs_strides,
            _lhs->get_shape(),
            _lhs->get_strides()
        )
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            rhs_strides,
            {},
            _rhs->get_strides()
        )
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            res_strides,
            {},
            _res->get_strides()
        )
    );
}

void AddAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->add(
        lhs, rhs, res,
        lhs_shape, lhs_strides,
        rhs_strides, res_strides
    );
}

std::string AddAction::to_string() const {
    std::ostringstream oss;
    oss << "AddAction: " << lhs->get_meta_info() << " + " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

AddEqAction::AddEqAction(Tensor* _lhs, const Tensor* _rhs)
    : Action(_lhs, _rhs, nullptr) {
    assert(_lhs->get_dim() == _rhs->get_dim());
    assert(_lhs->get_shape() == _rhs->get_shape());
    auto dim = _lhs->get_dim();
    lhs_shape = callocTensor(
        { dim },
        _lhs->get_name() + "_shape",
        INT32
    );
    lhs_strides = callocTensor(
        { dim },
        _lhs->get_name() + "_strides",
        INT32
    );
    rhs_strides = callocTensor(
        { dim },
        _rhs->get_name() + "_strides",
        INT32
    );
    gCreateAction(
        new AssignShapeAndStridesAction(
            lhs_shape,
            lhs_strides,
            _lhs->get_shape(),
            _lhs->get_strides()
        )
    );
    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            rhs_strides,
            {},
            _rhs->get_strides()
        )
    );
}

void AddEqAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(lhs->get_shape() == rhs->get_shape());
    g_backend_ops->addEq(lhs, rhs, lhs_shape, lhs_strides, rhs_strides);
}

std::string AddEqAction::to_string() const {
    std::ostringstream oss;
    oss << "AddEqAction: " << lhs->get_meta_info() << " += " << rhs->get_meta_info();
    return oss.str();
}

void ExpandAddAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->expandAdd(lhs, rhs, res);
}

std::string ExpandAddAction::to_string() const {
    std::ostringstream oss;
    oss << "ExpandAddAction: " << lhs->get_meta_info() << " + " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void ExpandMulAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->expandMul(lhs, rhs, res);
}

std::string ExpandMulAction::to_string() const {
    std::ostringstream oss;
    oss << "ExpandMulAction: " << lhs->get_meta_info() << " * " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void AtAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->at(lhs, rhs, res);
}

std::string AtAction::to_string() const {
    std::ostringstream oss;
    oss << "AtAction: " << lhs->get_meta_info() << " at " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

MulAction::MulAction(Tensor* _lhs, const Tensor* _rhs, Tensor* _res)
    : Action(_lhs, _rhs, _res) {
    assert(_lhs->get_dim() == _rhs->get_dim());
    assert(_lhs->get_shape() == _rhs->get_shape());
    auto dim = _lhs->get_dim();

    lhs_shape = callocTensor(
        { dim },
        _lhs->get_name() + "_shape",
        INT32
    );
    lhs_strides = callocTensor(
        { dim },
        _lhs->get_name() + "_strides",
        INT32
    );
    rhs_strides = callocTensor(
        { dim },
        _rhs->get_name() + "_strides",
        INT32
    );
    res_strides = callocTensor(
        { dim },
        _res->get_name() + "_strides",
        INT32
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            lhs_shape,
            lhs_strides,
            _lhs->get_shape(),
            _lhs->get_strides()
        )
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            rhs_strides,
            {},
            _rhs->get_strides()
        )
    );

    gCreateAction(
        new AssignShapeAndStridesAction(
            nullptr,
            res_strides,
            {},
            _res->get_strides()
        )
    );
}

void MulAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->mul(
        lhs, rhs, res,
        lhs_shape, lhs_strides,
        rhs_strides, res_strides
    );
}

std::string MulAction::to_string() const {
    std::ostringstream oss;
    oss << "MulAction: " << lhs->get_meta_info() << " * " << rhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void SumAction::execute() {
    assert(lhs != nullptr);
    assert(rhs == nullptr);
    if (dim < 0 || dim >= lhs->get_dim()) {
        std::cerr << "Error: Invalid dimension for sum operation" << std::endl;
        abort();
    }
    g_backend_ops->sum(lhs, res, dim);
}

std::string SumAction::to_string() const {
    std::ostringstream oss;
    oss << "SumAction: " << lhs->get_meta_info() << " -> " << res->get_meta_info() << " along dim " << dim;
    return oss.str();
}

void ReluAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->relu(lhs, res);
}

std::string ReluAction::to_string() const {
    std::ostringstream oss;
    oss << "ReluAction: " << lhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void ReluPrimeAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->reluPrime(lhs, res);
}

std::string ReluPrimeAction::to_string() const {
    std::ostringstream oss;
    oss << "ReluPrimeAction: " << lhs->get_meta_info() << " -> " << res->get_meta_info();
    return oss.str();
}

void CrossEntropyAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    g_backend_ops->crossEntropy(lhs, rhs, maxs, sums, res);
}

std::string CrossEntropyAction::to_string() const {
    std::ostringstream oss;
    oss << "CrossEntropyAction: " <<
        lhs->get_meta_info() << " with labels " <<
        rhs->get_meta_info() << " -> " <<
        res->get_meta_info() << " context : " <<
        maxs->get_meta_info() << ", " <<
        sums->get_meta_info();
    return oss.str();
}

void CrossEntropyBackwardAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(maxs != nullptr);
    assert(sums != nullptr);
    // lhs is zt
    // rhs is labels
    // res is grad
    g_backend_ops->crossEntropyBackward(lhs, rhs, maxs, sums, res);
}

std::string CrossEntropyBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "CrossEntropyBackwardAction: " <<
        lhs->get_meta_info() << " with labels " <<
        rhs->get_meta_info() << " -> " <<
        res->get_meta_info() << " context : " <<
        maxs->get_meta_info() << ", " <<
        sums->get_meta_info();
    return oss.str();
}

void CalcAllGradNormAction::execute() {
    assert(res != nullptr);
    g_backend_ops->calcAllGradNorm(grads, res);
}

std::string CalcAllGradNormAction::to_string() const {
    std::ostringstream oss;
    oss << "CalcAllGradNormAction: calculating gradient norm for " << grads.size() << " tensors" << " -> " << res->get_meta_info();
    return oss.str();
}

void ClipGradAction::execute() {
    assert(lhs != nullptr); // grad
    assert(rhs != nullptr); // norm
    g_backend_ops->clipGrad(lhs, rhs, grad_clip_val);
}

std::string ClipGradAction::to_string() const {
    std::ostringstream oss;
    oss << "ClipGradAction: clipping gradient " << lhs->get_meta_info() << " with norm " << rhs->get_meta_info() << " to grad_clip_val: " << grad_clip_val;
    return oss.str();
}

void AdamStepAction::execute() {
    param->inc_t();
    int t = param->get_t();
    Tensor* w = param->get_w();
    Tensor* grad = param->get_grad();
    Tensor* m = param->get_m();
    Tensor* v = param->get_v();

    g_backend_ops->adamStep(w, grad, m, v, t, lr, beta1, beta2, epsilon);
}

std::string AdamStepAction::to_string() const {
    std::ostringstream oss;
    oss << "AdamStepAction: updating parameter " << param->get_w()->get_meta_info() << " with learning rate " << lr;
    return oss.str();
}

void ZeroGradAction::execute() {
    g_backend_ops->memset(grad_tensors_data, 0, grad_tensors_data_capacity);
}

std::string ZeroGradAction::to_string() const {
    return "ZeroGradAction: zeroing gradients";
}

void ZeroCTensorsAction::execute() {
    g_backend_ops->memset(c_tensors_data, 0, c_tensors_data_capacity);
}

std::string ZeroCTensorsAction::to_string() const {
    return "ZeroCTensorsAction: zeroing c tensors";
}

void PrintNoZeroTensorNamesAction::execute() {
    for (const auto& tensor : g_c_tensors) {
        char* data = static_cast<char*>(::malloc(tensor->size()));
        g_backend_ops->cp_from_device(
            data,
            tensor,
            tensor->size()
        );
        bool succ = true;
        for (int i = 0; i < tensor->size(); ++i) {
            if (data[i] != (char)0) {
                succ = false;
                break;
            }
        }

        if (!succ) {
            std::cout << "Tensor with non-zero data: " << tensor->get_meta_info() << std::endl;
        }
        ::free(data);
    }
}

std::string PrintNoZeroTensorNamesAction::to_string() const {
    return "PrintNoZeroTensorNamesAction: printing tensors with non-zero data";
}

void FillWeightAction::execute() {
    assert(lhs != nullptr);
    if (init_type == "gauss") {
        g_backend_ops->init_weight_gauss(lhs, mean, sigma);
    }
    else if (init_type == "uniform") {
        g_backend_ops->init_weight_uniform(lhs, sigma);
    }
    else if (init_type == "xavier") {
        assert(false);
        // g_backend_ops->xavier(lhs);
    }
    else if (init_type == "kaiming") {
        assert(false);
        // g_backend_ops->kaiming(lhs);
    }
    else if (init_type == "dbg") {
        g_backend_ops->init_weight_for_dbg(lhs, sigma);
    }
    else if (init_type == "fill") {
        g_backend_ops->fill(lhs, sigma);
    }
    else {
        std::cerr << "Error: Unknown initialization type: " << init_type << std::endl;
        abort();
    }
}

std::string FillWeightAction::to_string() const {
    std::ostringstream oss;
    oss << "FillWeightAction: initializing " << lhs->get_meta_info()
        << " with type " << init_type
        << " sigma " << sigma
        << " mean " << mean;
    return oss.str();
}

std::string InitWeightAction::to_string() const {
    std::ostringstream oss;
    oss << "InitWeightAction: initializing " << lhs->get_meta_info()
        << " with type " << init_type
        << " sigma " << sigma
        << " mean " << mean;
    return oss.str();
}

void BoundaryAction::execute() {
    // Do nothing
}

std::string BoundaryAction::to_string() const {
    return "============= BoundaryAction: boundary action =============";
}

bool BoundaryAction::is_backward_boundary() const {
    return true;
}

AssignShapeAndStridesAction::AssignShapeAndStridesAction(
    Tensor* tensor_shape,
    Tensor* tensor_strides,
    const std::vector<int>& shape,
    const std::vector<int>& strides
) : Action(tensor_shape, nullptr, tensor_strides) {
    if (tensor_shape != nullptr) {
        shape_data = static_cast<int32_t*>(::malloc(sizeof(int32_t) * shape.size()));
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_data[i] = static_cast<int32_t>(shape[i]);
        }
    }
    if (tensor_strides != nullptr) {
        strides_data = static_cast<int32_t*>(::malloc(sizeof(int32_t) * strides.size()));
        for (size_t i = 0; i < strides.size(); ++i) {
            strides_data[i] = static_cast<int32_t>(strides[i]);
        }
    }
}

AssignShapeAndStridesAction::~AssignShapeAndStridesAction() {
    if (lhs != nullptr) {
        assert(shape_data != nullptr);
        ::free(shape_data);
    }
    if (res != nullptr) {
        assert(strides_data != nullptr);
        ::free(strides_data);
    }
}

void AssignShapeAndStridesAction::execute() {
    if (lhs != nullptr) {
        assert(shape_data != nullptr);
        g_backend_ops->cp_to_device(
            lhs,
            reinterpret_cast<char*>(shape_data),
            lhs->size()
        );
    }
    if (res != nullptr) {
        assert(strides_data != nullptr);
        g_backend_ops->cp_to_device(
            res,
            reinterpret_cast<char*>(strides_data),
            res->size()
        );
    }
}

std::string AssignShapeAndStridesAction::to_string() const {
    std::ostringstream oss;
    oss << "AssignShapeAndStridesAction: assigning shape " << (lhs == nullptr ? "null" : lhs->get_meta_info())
        << " and strides " << (res == nullptr ? "null" : res->get_meta_info());
    return oss.str();
}

AssignValueAction::AssignValueAction(Tensor* tensor, float value)
    : Action(tensor, nullptr, nullptr), value(value) {
    assert(tensor != nullptr);
    assert(tensor->get_dtype() == FLOAT32);
}

AssignValueAction::~AssignValueAction() {
    // No resources to free
}

void AssignValueAction::execute() {
    assert(lhs != nullptr);
    g_backend_ops->cp_to_device(
        lhs,
        reinterpret_cast<char*>(&value),
        sizeof(float)
    );
}

std::string AssignValueAction::to_string() const {
    std::ostringstream oss;
    oss << "AssignValueAction: assigning value " << value << " to " << lhs->get_meta_info();
    return oss.str();
}

void ReshapeDeepCpAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    g_backend_ops->reshape_deep_cp(lhs, rhs, shape, strides);
}

std::string ReshapeDeepCpAction::to_string() const {
    std::ostringstream oss;
    oss << "ReshapeDeepCpAction: deep copy " << lhs->get_meta_info() << " from " << rhs->get_meta_info() << " with strides " << strides->get_meta_info();
    return oss.str();
}

void RepeatInterleaveAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    assert(lhs->is_contiguous());
    assert(res->is_contiguous());
    g_backend_ops->repeat_interleave(lhs, res, n);
}

std::string RepeatInterleaveAction::to_string() const {
    std::ostringstream oss;
    oss << "RepeatInterleaveAction: repeat interleave " << lhs->get_meta_info() << " to " << res->get_meta_info() << " with n = " << n;
    return oss.str();
}

void SequenceMaskAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->sequence_mask(lhs, rhs, res, value);
}

std::string SequenceMaskAction::to_string() const {
    std::ostringstream oss;
    oss << "SequenceMaskAction: sequence mask " << lhs->get_meta_info() << " with mask " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void SoftmaxAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->softmax(lhs, res);
}

std::string SoftmaxAction::to_string() const {
    std::ostringstream oss;
    oss << "SoftmaxAction: softmax " << lhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void SoftmaxBackwardAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    // lhs is target grad
    // rhs is softmax result
    // res is input grad
    g_backend_ops->softmax_bacward(lhs, rhs, res);
}

std::string SoftmaxBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "SoftmaxBackwardAction: softmax backward " << res->get_meta_info() << " with softmax result " << rhs->get_meta_info() << " to " << lhs->get_meta_info();
    return oss.str();
}

void LazyDivAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    float fvalue = 0;
    g_backend_ops->div(res, lhs, value);
}

std::string LazyDivAction::to_string() const {
    std::ostringstream oss;
    oss << "LazyDivAction: lazy dividing " << lhs->get_meta_info() << " by " << value->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

DropoutMaskAction::DropoutMaskAction(Tensor* mask, float _p)
    : Action(nullptr, nullptr, mask), p(_p) {
    assert(mask != nullptr);
    shape = callocTensor(
        { mask->get_dim() },
        mask->get_name() + "_shape",
        INT32
    );
    strides = callocTensor(
        { mask->get_dim() },
        mask->get_name() + "_strides",
        INT32
    );
    gCreateAction(
        new AssignShapeAndStridesAction(
            shape,
            strides,
            mask->get_shape(),
            mask->get_strides()
        )
    );
}

void DropoutMaskAction::execute() {
    assert(res != nullptr);
    g_backend_ops->build_dropout_mask(res, p, shape, strides);
}

std::string DropoutMaskAction::to_string() const {
    std::ostringstream oss;
    oss << "Build Dropout mask: " << res->get_meta_info() << " with rate " << p;
    return oss.str();
}

void EmbeddingAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->embedding(lhs, rhs, res);
}

std::string EmbeddingAction::to_string() const {
    std::ostringstream oss;
    oss << "EmbeddingAction: embedding " << lhs->get_meta_info() << " with indices " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void EmbeddingBackwardAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->embeddingBackward(lhs, rhs, res);
}

std::string EmbeddingBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "EmbeddingBackwardAction: embedding backward " << lhs->get_meta_info() << " with indices " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void PosEncodingAction::execute() {
    assert(res != nullptr);
    g_backend_ops->pos_encoding(res);
}

std::string PosEncodingAction::to_string() const {
    std::ostringstream oss;
    auto shape = res->get_shape();
    oss << "PosEncodingAction: position encoding " << res->get_meta_info() << " with max_len " << shape[0] << " and num_hidden " << shape[1];
    return oss.str();
}

void AvgAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->avg(lhs, res);
}

std::string AvgAction::to_string() const {
    std::ostringstream oss;
    oss << "AvgAction: averaging " << lhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void NormAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    assert(src != nullptr);
    g_backend_ops->norm(src, lhs, rhs, res);
}

std::string NormAction::to_string() const {
    std::ostringstream oss;
    oss << "NormAction: normalizing " << src->get_meta_info() << " with mean " << lhs->get_meta_info() << " and variance " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void NormBackwardAction::execute() {
    assert(lhs != nullptr); // src grad
    assert(rhs != nullptr); // norm res
    assert(res != nullptr); // tgt grad
    assert(var_tensor != nullptr);
    const Tensor* src_grad = lhs;
    const Tensor* norm_res = rhs;
    Tensor* tgt_grad = res;
    g_backend_ops->normBackward(src_grad, norm_res, var_tensor, tgt_grad);
}

std::string NormBackwardAction::to_string() const {
    std::ostringstream oss;
    oss << "NormBackwardAction: normalizing backward " << lhs->get_meta_info()
        << " with norm res " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void DbgPrintAction::execute() {
    assert(lhs != nullptr);
    if (expected_name == "" || expected_name == lhs->get_name()) {
        std::cout << "[====== DEBUGGIN ======] " << msg << lhs->get_meta_info() << " : " << std::endl << *lhs << std::endl;
    }
}

std::string DbgPrintAction::to_string() const {
    std::ostringstream oss;
    oss << "DbgPrintAction: printing " << lhs->get_meta_info() << " with message " << msg;
    return oss.str();
}

void MemCpAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);

    g_backend_ops->cp_device_to_device(
        static_cast<char*>(lhs->get_data()) + offset_l,
        static_cast<char*>(rhs->get_data()) + offset_r,
        size
    );
}

std::string MemCpAction::to_string() const {
    std::ostringstream oss;
    oss << "MemCpAction: copying " << size << " bytes from " << rhs->get_meta_info() << " to " << lhs->get_meta_info() << " with offset " << offset_r << " to " << offset_l;
    return oss.str();
}

void VarAction::execute() {
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->var(lhs, rhs, res);
}

std::string VarAction::to_string() const {
    std::ostringstream oss;
    oss << "VarAction: variance " << lhs->get_meta_info() << " with mean " << rhs->get_meta_info() << " to " << res->get_meta_info();
    return oss.str();
}

void MulSVAction::execute() {
    assert(lhs != nullptr);
    assert(res != nullptr);
    g_backend_ops->mulSV(res, lhs, value);
}

std::string MulSVAction::to_string() const {
    std::ostringstream oss;
    oss << "MulSVAction: multiplying " << lhs->get_meta_info() << " with scalar " << value << " to " << res->get_meta_info();
    return oss.str();
}

void ClearAction::execute() {
    assert(lhs != nullptr);
    g_backend_ops->memset(lhs->get_data(), 0, lhs->size());
}

std::string ClearAction::to_string() const {
    std::ostringstream oss;
    oss << "ClearAction: clearing " << lhs->get_meta_info();
    return oss.str();
}

std::vector<Action*> g_actions;

std::vector<Action*> getOnceActions() {
    std::vector<Action*> once_actions;
    for (Action* action : g_actions) {
        if (action->is_do_once() && !action->executed_once()) {
            once_actions.push_back(action);
        }
    }
    return once_actions;
}

void gCreateAction(Action* action) {
    g_actions.push_back(action);
}

bool validateBoundaryFound() {
    bool boundary_found = false;
    for (Action* action : g_actions) {
        if (action->is_backward_boundary()) {
            boundary_found = true;
        }
    }
    return boundary_found;
}

bool validteZeroCTensorsFound() {
    assert(g_actions.size() > 0);
    return g_actions[0]->is_zero_c_tensors(); // the first action should be ZeroCTensorsAction
}

bool validateZeroGradFound() {
    /*
    zero grad action 一定要出现在第二个 非常重要！！！！
    因为我们的bmm实现中，会优先将结果切割成输入，这时候拷贝了上一次的grad，如果没有在最开始清除grad，就会有残留
    */
    assert(g_actions.size() > 1);
    return g_actions[1]->is_zero_grad(); // the second action should be ZeroGradAction
}

bool validateAddEqActionsInBackward() {
    // todo: all AddEqAction should be in backward(behind boundary)
    return true;
}

void gDoActions() {
    assert(validateBoundaryFound());
    assert(validateAddEqActionsInBackward());
    assert(validteZeroCTensorsFound());
    assert(validateZeroGradFound());
    g_training = true;
    g_backend_ops->prepare();
    for (Action* action : g_actions) {
        if (action->is_do_once() && action->executed_once()) {
            continue;
        }
        action->execute();
        action->increase_exec_times();
    }
    g_backend_ops->commit();
    g_backend_ops->wait();
}

void gDoOnceActions() {
    g_backend_ops->prepare();
    for (Action* action : g_actions) {
        if (!action->is_do_once() || action->executed_once()) {
            continue;
        }
        action->execute();
        action->increase_exec_times();
    }
    g_backend_ops->commit();
    g_backend_ops->wait();
}

void gDoForwardActions(bool training) {
    g_training = training;
    g_backend_ops->prepare();
    for (Action* action : g_actions) {
        if (action->is_do_once() && action->executed_once()) {
            continue;
        }
        if (action->is_backward_boundary()) {
            break;
        }
        action->execute();
        action->increase_exec_times();
    }
    g_backend_ops->commit();
    g_backend_ops->wait();
}

void gDoBackwardActions() {
    g_training = true;
    bool start = false;
    g_backend_ops->prepare();
    for (Action* action : g_actions) {
        if (action->is_do_once() && action->executed_once()) {
            continue;
        }
        if (action->is_backward_boundary()) {
            start = true;
        }
        if (!start) {
            continue; // skip actions before backward boundary
        }
        action->execute();
        action->increase_exec_times();
    }
    g_backend_ops->commit();
    g_backend_ops->wait();
}

void printAllActions() {
    std::cout << "Actions:" << std::endl;
    for (Action* action : g_actions) {
        if (action->is_do_once()) {
            std::cout << "[once]";
        }
        std::cout << *action << std::endl;
    }
}

void printDotGraph() {
    // save in out.dot 

    std::ofstream out("out.dot");
    out << "digraph G {" << std::endl;

    for (Tensor* tensor : g_tensors) {
        out << "Tensor_" << tensor->get_id() << " [shape=\"ellipse\" label=\"" << tensor->get_meta_info() << "\"];" << std::endl;
    }

    for (Tensor* tensor_view : g_tensor_views) {
        out << "Tensor_" << tensor_view->get_id() << " [shape=\"ellipse\" color=\"blue\" label=\"" << tensor_view->get_meta_info() << "\"];" << std::endl;
    }

    for (Tensor* c_tensor : g_c_tensors) {
        out << "Tensor_" << c_tensor->get_id() << " [shape=\"ellipse\" color=\"green\" label=\"" << c_tensor->get_meta_info() << "\"];" << std::endl;
    }

    for (Tensor* grad_tensor : g_grad_tensors) {
        out << "Tensor_" << grad_tensor->get_id() << " [shape=\"ellipse\" color=\"yellow\" label=\"" << grad_tensor->get_meta_info() << "\"];" << std::endl;
    }

    for (Action* action : g_actions) {
        out << "Action_" << action->get_id() << " [shape=\"box\" label=\"" << action->get_name() << "\"];" << std::endl;
    }

    for (Tensor* tensor_view : g_tensor_views) {
        // build edge
        auto parent = tensor_view->get_parent();
        if (parent != nullptr) {
            out << "Tensor_" << parent->get_id() << " -> Tensor_" << tensor_view->get_id() << ";" << std::endl;
        }
    }

    for (Action* action : g_actions) {
        std::string dot_string = action->get_dot_string();
        if (dot_string.empty()) {
            continue; // skip actions that do not have a dot string
        }
        out << dot_string;
    }

    out << "}" << std::endl;
}

void freeAllActions() {
    for (Action* action : g_actions) {
        delete action;
    }
    g_actions.clear();
}

void disableInitWeightAction() {
    for (Action* action : g_actions) {
        if (action->is_init_weight()) {
            action->increase_exec_times();
        }
    }
}

void disableOnceAction() {
    for (Action* action : g_actions) {
        if (action->is_do_once()) {
            action->increase_exec_times();
        }
    }
}
