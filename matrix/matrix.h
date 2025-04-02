#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <math.h>
#include <vector>
#include "stats/stats.h"

using namespace std;

#ifndef OMP_THREADS
#define OMP_THREADS 8
#endif

struct Shape {
    uint rowCnt;
    uint colCnt;
    Shape(uint r, uint c): rowCnt(r), colCnt(c) {}
    bool operator==(const Shape &s) {
        return rowCnt == s.rowCnt && colCnt == s.colCnt;
    }
    friend ostream &operator<<(ostream &output, const Shape &s) {
        output << "(" << s.rowCnt << ", " << s.colCnt << ")";
        return output;
    }
    uint size() {
        return rowCnt * colCnt;
    }
};

// typedef double DATATYPE;
typedef float DATATYPE;

class Matrix {

public:
    Matrix(Shape _shape);
    Matrix(const Matrix &m);
    ~Matrix();
    Matrix *zero();
    friend ostream &operator<<(ostream &output, const Matrix &m);
    Matrix *expand_add(const Matrix &m);
    Matrix *operator+(const Matrix &m);
    Matrix *operator+=(const Matrix &m);
    Matrix *operator+(DATATYPE dt);
    Matrix *operator-(DATATYPE dt);
    Matrix *operator-();
    Matrix *operator-(const Matrix &m);
    Matrix *operator-=(const Matrix &m);
    Matrix *operator*(const Matrix &m);
    Matrix *operator*(DATATYPE);
    Matrix *operator*=(DATATYPE);
    Matrix *operator/(DATATYPE);
    Matrix *Relu();
    Matrix *Relu_prime();
    Matrix *tanh();
    Matrix *tanh_prime();
    Matrix *sigmoid();
    Matrix *sigmoid_prime();
    Matrix& operator=(const Matrix &m);
    Matrix *pow2();
    friend Matrix *operator-(DATATYPE, const Matrix &m);
    DATATYPE* operator[](unsigned int index) const;
    Shape getShape() const;
    Matrix *at(const Matrix &m);
    Matrix *transpose();
    bool valid(uint x, uint y) const;
    void reShape(Shape shape);
    Matrix *assign(Matrix *);
    bool checkShape(const Matrix &m);
    Matrix *sum(uint dim);
    std::vector<Matrix *> split(uint dim);
    DATATYPE *getData() const;
    Matrix *fill(DATATYPE value);
    std::vector<uint> argMax();
    std::vector<DATATYPE> avg();
    std::vector<DATATYPE> var();
private:
    bool initialized;
    bool allocated;
    Shape shape;
    DATATYPE *data;
};

class TrainingData {
public:
    TrainingData(int, int);
    ~TrainingData();
    Matrix *x;
    uint y;
};

Matrix *sigmoid(const Matrix &m);
Matrix *sigmoid_prime(const Matrix &m);
Matrix *allocTmpMatrix(Matrix *m);
Matrix *allocTmpMatrix(const Matrix &m);
Matrix *allocTmpMatrix(const Shape & shape);
autograd::TmpMatricsStats tmpMatricsStats();
void freeTmpMatrix();
void init_weight(Matrix *weight, DATATYPE sigma, DATATYPE mean = 0);
void init_weight_uniform(Matrix *weight, DATATYPE sigma);

#endif