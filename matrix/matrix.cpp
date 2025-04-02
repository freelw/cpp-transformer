#include "matrix.h"

#include <iostream>
#include <assert.h>
#include <string.h>
#include <vector>
#include <omp.h> // Include OpenMP header
#include <random>
#include <chrono>
#include "stats/stats.h"

Matrix::Matrix(Shape _shape)
        : initialized(false),
        allocated(false),
        shape(_shape) {
    data = new DATATYPE[shape.size()];
    allocated = true;
    zero();
}

Matrix::Matrix(const Matrix &m):
    initialized(m.initialized),
    allocated(false),
    shape(m.shape) {
    assert(initialized);
    data = new DATATYPE[shape.size()];
    allocated = true;
    memcpy(data, m.data, sizeof(DATATYPE) * shape.rowCnt * shape.colCnt);
}

Matrix::~Matrix() {
    assert(initialized && allocated);
    delete [] data;
    data = nullptr;
}

Matrix *Matrix::zero() {
    assert(allocated);
    memset(data, 0, sizeof(DATATYPE) * shape.size());
    initialized = true;
    return this;
}

bool Matrix::checkShape(const Matrix &m) {
    if (!(this->getShape() == m.getShape())) {
        std::cerr << 
            "matrix shape missmatch." << 
            this->getShape() << " vs " << m.getShape()<< 
            std::endl;
        assert(false);
    }
    if (!m.initialized) {
        std::cerr << "matrix not initialized..." << std::endl;
        assert(false);
    }
    return true;
}

ostream &operator<<(ostream &output, const Matrix &m) {
    if (!m.initialized) {
        output << "matrix not initialized." << endl;
        return output;
    }
    output << "[";
    for (uint i = 0; i < m.shape.rowCnt; ++ i) {
        if (i > 0) {
            output << " ";
        }
        output << "[";
        for (uint j = 0; j < m.shape.colCnt-1; ++ j) {
            output << m[i][j] << ", ";
        }
        output << m[i][m.shape.colCnt-1] << "]";
        if (i < m.shape.rowCnt-1) {
            output << endl;
        }
    }
    output << "]" << endl;
    return output;
}

Matrix *Matrix::expand_add(const Matrix &m) {
    assert(m.shape.rowCnt == shape.rowCnt);
    assert(m.shape.colCnt == 1);
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] += m[i][0];
        }
    }
    return res;
}

Matrix *Matrix::operator+(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; ++i) {
        DATATYPE *m_data = m[i];
        DATATYPE *this_data = (*res)[i];
        for (uint j = 0; j < shape.colCnt; ++j) {
            this_data[j] += m_data[j];
        }
    }
    return res;
}

Matrix *Matrix::operator+=(const Matrix &m) {
    checkShape(m);
    DATATYPE *m_data = m.getData();
    DATATYPE *this_data = this->getData();

    const uint blockSize = 16; // Block size for cache optimization
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; i += blockSize) {
        for (uint j = 0; j < shape.colCnt; j += blockSize) {
            for (uint ii = i; ii < std::min(i + blockSize, shape.rowCnt); ++ii) {
                for (uint jj = j; jj < std::min(j + blockSize, shape.colCnt); ++jj) {
                    this_data[ii * shape.colCnt + jj] += m_data[ii * shape.colCnt + jj];
                }
            }
        }
    }
    return this;
}

Matrix *Matrix::pow2() {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*res)[i][j];
            r = std::pow(r, 2);
        }
    }
    return res;
}

Matrix *Matrix::operator+(DATATYPE dt) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] += dt;
        }
    }
    return res;
}

Matrix *Matrix::operator-(DATATYPE dt) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] -= dt;
        }
    }
    return res;
}

Matrix *Matrix::operator-() {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            auto &r = (*res)[i][j];
            r = -r;
        }
    }
    return res;
}

Matrix *operator-(DATATYPE v, const Matrix &m) {
    Matrix *res = allocTmpMatrix(m);
    for (uint i = 0; i < m.shape.rowCnt; ++i) {
        for (uint j = 0; j < m.shape.colCnt; ++j) {
            auto &r = (*res)[i][j];
            r = v-r;
        }
    }
    return res;
}

Matrix *Matrix::operator-(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] -= m[i][j];
        }
    }
    return res;
}

Matrix *Matrix::operator-= (const Matrix &m) {
    checkShape(m);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*this)[i][j] -= m[i][j];
        }
    }
    return this;
}

Matrix *Matrix::operator*(const Matrix &m) {
    checkShape(m);
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] *= m[i][j];
        }
    }
    return res;
}

Matrix *Matrix::operator*(DATATYPE v) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] *= v;
        }
    }
    return res;
}

Matrix *Matrix::operator*=(DATATYPE v) {
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*this)[i][j] *= v;
        }
    }
    return this;
}

Matrix *Matrix::operator/(DATATYPE v) {
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] /= v;
        }
    }
    return res;
}

Matrix *Matrix::Relu() {
    Matrix *res = allocTmpMatrix(this);
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; ++i) {
        auto row = (*res)[i];
        for (uint j = 0; j < shape.colCnt; ++j) {
            row[j] = std::max(row[j], (DATATYPE)0);
        }
    }
    return res;
}

Matrix *Matrix::Relu_prime() {
    Matrix *res = allocTmpMatrix(this);
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; ++i) {
        auto row = (*res)[i];
        for (uint j = 0; j < shape.colCnt; ++j) {
            row[j] = row[j] > 0 ? 1 : 0;
        }
    }
    return res;
}

Matrix *Matrix::tanh() {
    Matrix *res = allocTmpMatrix(this);
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; ++i) {
        auto row = (*res)[i];
        for (uint j = 0; j < shape.colCnt; ++j) {
            row[j] = std::tanh(row[j]);
        }
    }
    return res;
}

Matrix *Matrix::tanh_prime() {
    return 1 - *(this->tanh()->pow2());
}

Matrix& Matrix::operator=(const Matrix &m) {
    assert(m.initialized);
    this->reShape(m.shape);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*this)[i][j] = m[i][j];
        }
    }
    return *this;
}

DATATYPE *Matrix::operator[](unsigned int index) const {
    assert(index < shape.rowCnt);
    return (DATATYPE *)&(data[index*shape.colCnt]);
}

Shape Matrix::getShape() const {
    return shape;
}

Matrix *Matrix::at(const Matrix &m) {
    assert(m.shape.rowCnt == shape.colCnt);
    Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, m.shape.colCnt));

    // DATATYPE *data = res->getData();
    // DATATYPE *m_data = m.getData();
    // DATATYPE *this_data = this->getData();

    // // use openmp for parallelization
    // #pragma omp parallel for collapse(2) num_threads(OMP_THREADS)
    // for (uint i = 0; i < shape.rowCnt; ++i) {
    //     for (uint j = 0; j < m.shape.colCnt; ++j) {
    //         DATATYPE sum = 0;
    //         for (uint k = 0; k < shape.colCnt; ++k) {
    //             sum += this_data[i * shape.colCnt + k] * m_data[k * m.shape.colCnt + j];
    //         }
    //         data[i * m.shape.colCnt + j] = sum;
    //     }
    // }

    DATATYPE *A = this->getData();
    DATATYPE *B = m.getData();
    DATATYPE *C = res->getData();

    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint k = 0; k < shape.colCnt; ++k) {
            for (uint j = 0; j < m.shape.colCnt; ++j) {
                C[i * m.shape.colCnt + j] += A[i * shape.colCnt + k] * B[k * m.shape.colCnt + j];
            }
        }
    }

    return res;
}

Matrix *Matrix::transpose() {
    Matrix *res = allocTmpMatrix(Shape(shape.colCnt, shape.rowCnt));
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE *data = (*res)[i];
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            data[j] = (*this)[j][i];
        }
    }
    return res;
}

bool Matrix::valid(uint x, uint y) const {
    return allocated && initialized && x < shape.rowCnt && y < shape.colCnt;
}

void Matrix::reShape(Shape _shape) {
    assert(allocated && initialized);
    delete []data;
    shape = _shape;
    data = new DATATYPE[shape.size()];
    zero();
}

Matrix *Matrix::assign(Matrix *other) {
    assert(allocated && initialized);
    checkShape(other->getShape());
    memcpy(data, other->data, sizeof(DATATYPE) * shape.size());
    return this;
}

Matrix *Matrix::sum(uint dim) {
    assert(dim == 1);
    if (dim == 1) {
        Matrix *res = allocTmpMatrix(Shape(shape.rowCnt, 1));
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                (*res)[i][0] += (*this)[i][j];
            }
        }
        return res;
    }
    return nullptr;
}

std::vector<Matrix *> Matrix::split(uint dim) {
    assert(dim == 1);
    if (dim == 1) {
        std::vector<Matrix *> res;
        for (uint i = 0; i < shape.colCnt; ++ i) {
            Matrix *m = allocTmpMatrix(Shape(shape.rowCnt, 1));
            for (uint j = 0; j < shape.rowCnt; ++ j) {
                (*m)[j][0] = (*this)[j][i];
            }
            res.push_back(m);
        }
        return res;
    }
    return {};
}

DATATYPE *Matrix::getData() const {
    return data;
}

Matrix *Matrix::fill(DATATYPE value) {
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            (*this)[i][j] = value;
        }
    }
    return this;
}

std::vector<uint> Matrix::argMax() {
    Shape shape = getShape();
    std::vector<uint> res;
    res.reserve(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        uint max_index = 0;
        for (uint j = 1; j < shape.rowCnt; ++ j) {
            if ((*this)[j][i] > (*this)[max_index][i]) {
                max_index = j;
            }
        }
        res.push_back(max_index);
    }
    return res;
}

std::vector<DATATYPE> Matrix::avg() {
    Shape shape = getShape();
    std::vector<DATATYPE> res;
    res.reserve(shape.colCnt);
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE sum = 0;
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            sum += (*this)[j][i];
        }
        res.push_back(sum/shape.rowCnt);
    }
    return res;
}

std::vector<DATATYPE> Matrix::var() {
    std::vector<DATATYPE> res;
    std::vector<DATATYPE> avg_res = this->avg();
    Shape shape = getShape();
    for (uint i = 0; i < shape.colCnt; ++ i) {
        DATATYPE sum = 0;
        auto avg_r = avg_res[i];
        for (uint j = 0; j < shape.rowCnt; ++ j) {
            sum += std::pow(((*this)[j][i] - avg_r), 2);
        }
        res.push_back(sum/shape.rowCnt);
    }
    return res;
}

DATATYPE _sigmoid(DATATYPE z) {
    return 1./(1.+exp(-z));
}

Matrix *Matrix::sigmoid() {
    Shape shape = getShape();
    Matrix *res = allocTmpMatrix(this);
    for (uint i = 0; i < shape.rowCnt; ++i) {
        for (uint j = 0; j < shape.colCnt; ++j) {
            (*res)[i][j] = _sigmoid((*res)[i][j]);
        }
    }
    return res;
}

Matrix *Matrix::sigmoid_prime() {
    return *sigmoid() * *(1 - *sigmoid());
}

std::vector<Matrix *> tmpMatrics;
Matrix *allocTmpMatrix(Matrix *m) {
    return allocTmpMatrix(*m);
}

Matrix *allocTmpMatrix(const Matrix &m) {
    Matrix *res = new Matrix(m);
    tmpMatrics.push_back(res);
    return res;
}

Matrix *allocTmpMatrix(const Shape & shape) {
    Matrix *res = new Matrix(shape);
    res->zero();
    tmpMatrics.push_back(res);
    return res;
}

autograd::TmpMatricsStats tmpMatricsStats() {
    autograd::TmpMatricsStats stats;

    uint size = tmpMatrics.size();
    uint bytes = 0;
    for (auto p : tmpMatrics) {
        bytes += p->getShape().size() * sizeof(DATATYPE);
    }
    stats.size = size;
    stats.bytes = bytes;
    return stats;
}

void freeTmpMatrix() {
    for (auto p : tmpMatrics) {
        delete p;
    }
    tmpMatrics.clear();
}

TrainingData::TrainingData(int input_layer_size, int _y)
    : y(_y) {  
    x = new Matrix(Shape(input_layer_size, 1));
    x->zero();
}

TrainingData::~TrainingData() {
    delete x;
}

void init_weight(Matrix *weight, DATATYPE sigma, DATATYPE mean) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<DATATYPE> distribution_w(0.0, sigma);
    for (uint i = 0; i < weight->getShape().rowCnt; ++ i) {
        for (uint j = 0; j < weight->getShape().colCnt; ++ j) {
            (*weight)[i][j] = distribution_w(generator_w) + mean;
        }
    }
}

void init_weight_uniform(Matrix *weight, DATATYPE sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::uniform_real_distribution<DATATYPE> distribution_w(-sigma, sigma);
    for (uint i = 0; i < weight->getShape().rowCnt; ++ i) {
        for (uint j = 0; j < weight->getShape().colCnt; ++ j) {
            (*weight)[i][j] = distribution_w(generator_w);
        }
    }
}