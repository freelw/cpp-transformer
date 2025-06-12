#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cassert>
#include <ostream>
#include <string>

#define TENSOR_PADDING_SIZE 0

enum TensorDType {
    INT8 = 0,
    INT16 = 1,
    INT32,
    INT64,
    FLOAT16,
    FLOAT32,
    FLOAT64
};

std::string TensorDtype_to_string(TensorDType dtype);

struct TensorStorage {
    TensorStorage(int _size) : data(nullptr), size(_size) {}
    void* data;
    int size;
};

extern int g_tensor_id;

class Tensor {
public:
    Tensor(const std::vector<int>& _shape, const std::string& _name, TensorDType _dtype);
    Tensor(const std::vector<int>& _shape, TensorDType _dtype);
    Tensor(
        const std::vector<int>& _shape, const std::vector<int>& _strides,
        const std::string& _name, TensorDType _dtype, TensorStorage* _storage,
        int _offset
    );
    Tensor(
        const std::vector<int>& _shape, const std::vector<int>& _strides,
        const std::string& _name, TensorDType _dtype, TensorStorage* _storage
    );
    virtual ~Tensor();
    virtual void set_data(void* ptr);
    virtual void* get_data() const;
    TensorStorage* get_storage() const { return storage; }
    int get_offset() const { return offset; }
    virtual int size() const;
    virtual int length() const;
    virtual int capacity() const;
    virtual bool is_view() const { return !own_storage; }
    std::vector<int> get_shape() const { return shape; }
    std::vector<int> get_strides() const { return strides; }
    virtual int get_dim() const { return shape.size(); }
    TensorDType get_dtype() const { return dtype; }
    virtual std::string get_name() const { return name; }
    Tensor* transpose(int a = 0, int b = 1);
    Tensor* permute(const std::vector<int>& dims);
    Tensor* reshape(const std::vector<int>& shape) const;
    Tensor* fill(float value);
    Tensor* repeat_interleave(int n);
    Tensor* sequence_mask(Tensor* mask, float value);
    Tensor* softmax();
    std::string get_meta_info() const;
    bool is_contiguous() const;
    bool is_shared_with(const Tensor* other) const {
        return this->get_storage() == other->get_storage();
    }
    friend std::ostream& operator<<(std::ostream& output, const Tensor& s);
    int get_id() const { return id; }
protected:
    int cell_size() const;
    int gen_id() {
        return ++g_tensor_id;
    }
protected:
    std::vector<int> shape;
    std::vector<int> strides;
    std::string name;
    TensorDType dtype;
private:
    const bool own_storage;
    TensorStorage* storage;
    int offset;
    int id;
};

extern std::vector<Tensor*> g_tensors;
extern std::vector<Tensor*> g_c_tensors;
extern std::vector<Tensor*> g_tensor_views;
extern std::vector<Tensor*> g_grad_tensors;
extern size_t tensors_data_capacity;
extern size_t c_tensors_data_capacity;
extern size_t grad_tensors_data_capacity;

Tensor* allocTensor(const std::vector<int>& shape, const std::string& name, TensorDType _dtype = FLOAT32);
Tensor* callocTensor(const std::vector<int>& shape, const std::string& name, TensorDType _dtype = FLOAT32);
Tensor* allocTensor(const std::vector<int>& shape, TensorDType _dtype = FLOAT32);
Tensor* allocTensorView(
    const Tensor* parent, const std::vector<int>& shape,
    const std::vector<int>& strides, const std::string& name,
    int offset = 0);
Tensor* allocGradTensor(const std::vector<int>& shape, const std::string& name);
Tensor* allocGradTensor(const std::vector<int>& shape);
void printAllTensors();

void freeAllTensors();
void freeAllCTensors();
void freeAllTensorViews();
void freeAllGradTensors();
void validateAllTensors(); // for dbg
void validateAllTensorNames(); // for dbg

extern void* grad_tensors_data;
extern size_t grad_tensors_data_capacity;
extern void* c_tensors_data;
extern size_t c_tensors_data_capacity;

void allocMemAndInitTensors();
void releaseTensorMem();

#endif