#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/actions.h"
#include "graph/node.h"
#include <sstream>
#include <cmath>

std::string TensorDtype_to_string(TensorDType dtype) {
    switch (dtype) {
    case INT8: return "INT8";
    case INT16: return "INT16";
    case INT32: return "INT32";
    case INT64: return "INT64";
    case FLOAT16: return "FLOAT16";
    case FLOAT32: return "FLOAT32";
    case FLOAT64: return "FLOAT64";
    default: return "UNKNOWN";
    }
}

Tensor::Tensor(const std::vector<int>& _shape, const std::string& _name, TensorDType _dtype)
    : shape(_shape), name(_name), dtype(_dtype), own_storage(true), offset(0), id(0) {
    strides.resize(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    storage = new TensorStorage(size());
    assert(id == 0);
    id = gen_id();
}

Tensor::Tensor(const std::vector<int>& _shape, TensorDType _dtype)
    : Tensor(_shape, "", _dtype) {

}

Tensor::Tensor(
    const std::vector<int>& _shape,
    const std::vector<int>& _strides,
    const std::string& _name,
    TensorDType _dtype,
    TensorStorage* _storage,
    int _offset)
    : shape(_shape), strides(_strides),
    name(_name), dtype(_dtype),
    own_storage(false), storage(_storage),
    offset(_offset), id(0) {
    assert(shape.size() == strides.size());
    assert(_storage != nullptr);
    assert(_offset >= 0);
    assert(_offset < _storage->size);
    assert(id == 0);
    id = gen_id();
}

Tensor::Tensor(
    const std::vector<int>& _shape,
    const std::vector<int>& _strides,
    const std::string& _name,
    TensorDType _dtype,
    TensorStorage* _storage)
    : Tensor(
        _shape, _strides, _name, _dtype, _storage, 0
    ) {
    assert(shape.size() == strides.size());
}

Tensor::~Tensor() {
    if (own_storage) {
        delete storage;
    }
}

void Tensor::set_data(void* ptr, void* ctx) {
    assert(ptr != nullptr);
    assert(storage != nullptr);
    assert(storage->data == nullptr);
    storage->data = ptr;
    storage->ctx = ctx;
}

void* Tensor::get_data() const {
    assert(storage != nullptr);
    assert(storage->data != nullptr);
    return static_cast<char*>(storage->data) + offset * cell_size();
}

int Tensor::size() const {
    return length() * cell_size();
}

int Tensor::length() const {
    int total_length = 1;
    for (int dim : shape) {
        total_length *= dim;
    }
    return total_length;
}

int Tensor::cell_size() const {
    switch (dtype) {
    case INT8: return 1;
    case INT16: return 2;
    case INT32: return 4;
    case INT64: return 8;
    case FLOAT16: return 2;
    case FLOAT32: return 4;
    case FLOAT64: return 8;
    default: assert(false); return 0;
    }
}

int Tensor::capacity() const {
    return size() + TENSOR_PADDING_SIZE;
}

Tensor* Tensor::transpose(int a, int b) {
    auto strides = this->get_strides();
    auto shape = this->get_shape();

    assert(a < shape.size());
    assert(b < shape.size());
    assert(a != b);

    std::swap(strides[a], strides[b]);
    std::swap(shape[a], shape[b]);
    Tensor* transpose_view = allocTensorView(
        this,
        shape,
        strides,
        this->get_name() + "_transpose"
    );
    return transpose_view;
}

Tensor* Tensor::permute(const std::vector<int>& dims) {
    assert(dims.size() == get_dim());
    std::vector<int> new_shape(get_dim());
    std::vector<int> new_strides(get_dim());
    for (int i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape[dims[i]];
        new_strides[i] = strides[dims[i]];
    }
    Tensor* permute_view = allocTensorView(
        this,
        new_shape,
        new_strides,
        this->get_name() + "_permute"
    );
    return permute_view;
}

Tensor* Tensor::reshape(const std::vector<int>& shape) const {
    std::vector<int> calc_req_shape = shape;

    int unknown_dim_cnt = 0;
    int unknown_dim_index = -1;
    for (int i = 0; i < calc_req_shape.size(); ++i) {
        if (calc_req_shape[i] == -1) {
            unknown_dim_cnt++;
            unknown_dim_index = i;
        }
    }

    assert(unknown_dim_cnt <= 1);

    if (unknown_dim_cnt == 1) {
        int total_length = 1;
        for (int i = 0; i < calc_req_shape.size(); ++i) {
            if (calc_req_shape[i] != -1) {
                total_length *= calc_req_shape[i];
            }
        }
        assert(length() % total_length == 0);
        calc_req_shape[unknown_dim_index] = length() / total_length;
    }

    auto req_length = 1;
    for (int i = 0; i < calc_req_shape.size(); ++i) {
        req_length *= calc_req_shape[i];
    }

    assert(req_length == length());

    if (this->is_contiguous()) {
        std::vector<int> new_strides(calc_req_shape.size());
        new_strides[calc_req_shape.size() - 1] = 1;
        for (int i = calc_req_shape.size() - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * calc_req_shape[i + 1];
        }
        Tensor* reshape_view = allocTensorView(
            this,
            calc_req_shape,
            new_strides,
            this->get_name() + "_reshape"
        );
        return reshape_view;
    }
    else {
        Tensor* reshape_deep_cpy = callocTensor(
            calc_req_shape,
            this->get_name() + "_reshape_deep_copy",
            this->get_dtype()
        );
        Tensor* tensor_shape = callocTensor(
            { get_dim() },
            this->get_name() + "_reshape_deep_copy_shape",
            INT32
        );
        Tensor* tensor_strides = callocTensor(
            { get_dim() },
            this->get_name() + "_reshape_deep_copy_strides",
            INT32
        );

        gCreateAction(
            new AssignShapeAndStridesAction(
                tensor_shape,
                tensor_strides,
                this->get_shape(),
                this->get_strides()
            )
        );

        gCreateAction(
            new ReshapeDeepCpAction(
                reshape_deep_cpy,
                this,
                tensor_shape,
                tensor_strides
            )
        );

        return reshape_deep_cpy;
    }
}

Tensor* Tensor::fill(float value) {
    // assert(!is_view());
    assert(dtype == FLOAT32);
    g_backend_ops->fill(this, value);
    return this;
}

Tensor* Tensor::repeat_interleave(int n) {
    assert(!is_view());
    assert(dtype == INT32);
    auto dim = get_dim();
    std::vector<int> new_shape = shape;
    if (dim == 1) {
        new_shape[0] *= n;
    }
    else {
        new_shape[dim - 2] *= n;
    }
    Tensor* repeat_interleave_tensor = callocTensor(
        new_shape,
        this->get_name() + "_repeat_interleave",
        INT32
    );
    gCreateAction(
        new RepeatInterleaveAction(
            this,
            repeat_interleave_tensor,
            n
        )
    );
    return repeat_interleave_tensor;
}

Tensor* Tensor::sequence_mask(Tensor* mask, float value) {
    assert(mask->get_dtype() == INT32);
    assert(mask->get_dim() == 1);
    assert(mask->get_shape()[0] == shape[0]);
    assert(this->get_dtype() == FLOAT32);
    assert(this->get_dim() == 2);
    Tensor* sequence_mask_tensor = callocTensor(
        { shape[0], shape[1] },
        this->get_name() + "_sequence_mask",
        this->get_dtype()
    );
    gCreateAction(
        new SequenceMaskAction(
            this,
            mask,
            sequence_mask_tensor,
            value
        )
    );
    return sequence_mask_tensor;
}

Tensor* Tensor::softmax() {
    Tensor* res = callocTensor(
        shape,
        this->get_name() + "_softmax_res",
        this->get_dtype()
    );
    gCreateAction(
        new SoftmaxAction(
            this,
            res
        )
    );
    return res;
}

std::string Tensor::get_meta_info() const {
    std::ostringstream output;
    output << "Tensor[" << get_id() << "]";
    if (get_dtype() != FLOAT32) {
        std::string dtype_str = TensorDtype_to_string(get_dtype());
        output << "(" << dtype_str << ")";
    }
    output << "(" << get_name() << ")(";
    for (size_t i = 0; i < shape.size(); ++i) {
        output << shape[i];
        if (i != shape.size() - 1) {
            output << ", ";
        }
    }
    output << ")";
    return output.str();
}

bool Tensor::is_contiguous() const {
    auto dim = get_dim();
    if (strides[dim - 1] != 1) {
        return false;
    }
    for (int i = 0; i < dim - 1; ++i) {
        if (strides[i] != strides[i + 1] * shape[i + 1]) {
            return false;
        }
    }
    return true;
}

void dfs_print(
    std::ostream& output, const Tensor& s,
    void* data, int depth,
    int base_offset, bool is_start) {
    if (!is_start) {
        for (int i = 0; i < depth; ++i) {
            output << " ";
        }
    }
    output << "[";
    auto dim = s.get_dim();
    auto stride = s.get_strides()[depth];
    if (depth == dim - 1) {
        auto dtype = s.get_dtype();
        auto length = s.get_shape()[dim - 1];
        for (int i = 0; i < length; ++i) {
            if (dtype == FLOAT32) {
                output << *(reinterpret_cast<float*>(data) + base_offset + i * stride);
            }
            else if (dtype == INT32) {
                output << *(reinterpret_cast<int32_t*>(data) + base_offset + i * stride);
            }
            if (i < length - 1) {
                output << ", ";
            }
            else {
                output << "]";
            }
        }
        return;
    }
    for (int i = 0; i < s.get_shape()[depth]; ++i) {
        dfs_print(output, s, data, depth + 1, base_offset + i * stride, i == 0);
        if (i < s.get_shape()[depth] - 1) {
            output << ",";
            for (int j = 0; j < dim - depth - 1; ++j) {
                output << std::endl;
            }
        }
    }
    output << "]";
}

std::ostream& operator<<(std::ostream& output, const Tensor& s) {
    void* data = ::malloc(s.size());
    g_backend_ops->cp_from_device(
        reinterpret_cast<char*>(data),
        &s,
        s.size()
    );
    dfs_print(output, s, data, 0, 0, true);
    ::free(data);
    return output;
}

std::vector<Tensor*> g_tensors;
std::vector<Tensor*> g_tensor_views;
std::vector<Tensor*> g_grad_tensors;
std::vector<Tensor*> g_c_tensors; // temp tensors should be clear in each epoch

Tensor* allocTensor(const std::vector<int>& shape, const std::string& name, TensorDType dtype) {
    Tensor* tensor = new Tensor(shape, name, dtype);
    g_tensors.push_back(tensor);
    return tensor;
}

Tensor* callocTensor(const std::vector<int>& shape, const std::string& name, TensorDType dtype) {
    Tensor* tensor = new Tensor(shape, name, dtype);
    g_c_tensors.push_back(tensor);
    gCreateAction(
        new ClearAction(
            tensor
        )
    );
    return tensor;
}

Tensor* allocTensor(const std::vector<int>& shape, TensorDType dtype) {
    return allocTensor(shape, "tensor_autoname", dtype);
}

Tensor* allocTensorView(
    const Tensor* parent, const std::vector<int>& shape,
    const std::vector<int>& strides, const std::string& name,
    int offset
) {
    Tensor* tensor_view = new Tensor(
        shape, strides, name,
        parent->get_dtype(), parent->get_storage(),
        parent->get_offset() + offset
    );
    g_tensor_views.push_back(tensor_view);
    return tensor_view;
}

Tensor* allocGradTensor(const std::vector<int>& shape, const std::string& name) {
    Tensor* grad_tensor = new Tensor(shape, name, FLOAT32);
    g_grad_tensors.push_back(grad_tensor);
    return grad_tensor;
}

Tensor* allocGradTensor(const std::vector<int>& shape) {
    return allocGradTensor(shape, "grad_autoname");
}

void printAllTensors() {
    std::cout << "Tensors:" << std::endl;
    for (Tensor* tensor : g_tensors) {
        std::cout << "\t" << tensor->get_meta_info() << std::endl;
    }
    std::cout << "Tensor Views:" << std::endl;
    for (Tensor* tensor_view : g_tensor_views) {
        std::cout << "\t" << tensor_view->get_meta_info() << std::endl;
    }
    std::cout << "Grad Tensors:" << std::endl;
    for (Tensor* grad_tensor : g_grad_tensors) {
        std::cout << "\t" << grad_tensor->get_meta_info() << std::endl;
    }
}

void freeAllTensors() {
    for (Tensor* tensor : g_tensors) {
        delete tensor;
    }
    g_tensors.clear();
    g_tensor_id = 0;
}

void freeAllCTensors() {
    for (Tensor* c_tensor : g_c_tensors) {
        delete c_tensor;
    }
    g_c_tensors.clear();
}

void freeAllTensorViews() {
    for (Tensor* tensor_view : g_tensor_views) {
        delete tensor_view;
    }
    g_tensor_views.clear();
}

void freeAllGradTensors() {
    for (Tensor* grad_tensor : g_grad_tensors) {
        delete grad_tensor;
    }
    g_grad_tensors.clear();
}


void validateAllTensors() {
    for (Tensor* tensor : g_tensors) {
        char* buffer = reinterpret_cast<char*>(::malloc(tensor->size()));

        g_backend_ops->cp_from_device(
            buffer,
            tensor,
            tensor->size()
        );

        if (tensor->get_dtype() == FLOAT32) {
            float* data = reinterpret_cast<float*>(buffer);
            for (int i = 0; i < tensor->length(); ++i) {
                bool valid = !std::isnan(data[i]) && !std::isinf(data[i]);
                if (!valid) {
                    std::cerr << "Invalid value at index " << i << " in tensor "
                        << tensor->get_meta_info() << ": " << data[i] << std::endl;
                }
            }
        }
        else if (tensor->get_dtype() == INT32) {
            int32_t* data = reinterpret_cast<int32_t*>(buffer);
            for (int i = 0; i < tensor->length(); ++i) {
                bool valid = !std::isnan(data[i]) && !std::isinf(data[i]);
                if (!valid) {
                    std::cerr << "Invalid value at index " << i << " in tensor "
                        << tensor->get_meta_info() << ": " << data[i] << std::endl;
                }
            }
        }
        ::free(buffer);
    }
}

void validateAllTensorNames() {
    for (Tensor* tensor : g_tensors) {
        auto name = tensor->get_name();
        if (name.find("grad") != std::string::npos) {
            std::cerr << "Tensor name contains 'grad': " << name << std::endl;
            abort();
        }
    }
}

void* tensors_data = nullptr;
void* c_tensors_data = nullptr;
void* grad_tensors_data = nullptr;
size_t tensors_data_capacity = 0;
size_t c_tensors_data_capacity = 0;
size_t grad_tensors_data_capacity = 0;

void allocMemAndInitTensors() {

    graph::validateAllNodes();
    assert(tensors_data == nullptr);
    assert(grad_tensors_data == nullptr);
    assert(tensors_data_capacity == 0);
    assert(grad_tensors_data_capacity == 0);

    for (Tensor* tensor : g_tensors) {
        tensors_data_capacity += tensor->capacity();
    }
    for (Tensor* tensor : g_c_tensors) {
        c_tensors_data_capacity += tensor->capacity();
    }
    for (Tensor* tensor : g_grad_tensors) {
        grad_tensors_data_capacity += tensor->capacity();
    }

    void* tensors_data_ctx = nullptr;
    void* c_tensors_data_ctx = nullptr;
    void* grad_tensors_data_ctx = nullptr;
    tensors_data = g_backend_ops->alloc(tensors_data_capacity, &tensors_data_ctx);
    c_tensors_data = g_backend_ops->alloc(c_tensors_data_capacity, &c_tensors_data_ctx);
    grad_tensors_data = g_backend_ops->alloc(grad_tensors_data_capacity, &grad_tensors_data_ctx);

    g_backend_ops->memset(tensors_data, 0, tensors_data_capacity);
    g_backend_ops->memset(c_tensors_data, 0, c_tensors_data_capacity);
    g_backend_ops->memset(grad_tensors_data, 0, grad_tensors_data_capacity);

    int64_t offset = 0;
    for (Tensor* tensor : g_tensors) {
        tensor->set_data(reinterpret_cast<char*>(tensors_data) + offset, tensors_data_ctx);
        offset += tensor->capacity();
    }

    offset = 0;
    for (Tensor* tensor : g_c_tensors) {
        tensor->set_data(reinterpret_cast<char*>(c_tensors_data) + offset, c_tensors_data_ctx);
        offset += tensor->capacity();
    }

    offset = 0;
    for (Tensor* tensor : g_grad_tensors) {
        tensor->set_data(reinterpret_cast<char*>(grad_tensors_data) + offset, grad_tensors_data_ctx);
        offset += tensor->capacity();
    }
}

void releaseTensorMem() {
    assert(tensors_data != nullptr);
    if (grad_tensors_data != nullptr) {
        g_backend_ops->free(grad_tensors_data);
        grad_tensors_data = nullptr;
    }
    if (c_tensors_data != nullptr) {
        g_backend_ops->free(c_tensors_data);
        c_tensors_data = nullptr;
    }
    g_backend_ops->free(tensors_data);
    tensors_data = nullptr;
    tensors_data_capacity = 0;
    c_tensors_data_capacity = 0;
    grad_tensors_data_capacity = 0;
}

int g_tensor_id = 0;