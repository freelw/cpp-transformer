#include <string.h>
#include <iostream>

#include "autograd/parameters.h"

namespace autograd {

    Parameters::Parameters(Node *node) {
        w = node;
        m = new Matrix(w->getShape());
        v = new Matrix(w->getShape());
        t = 0;
    }

    Parameters::~Parameters() {
        delete m;
        delete v;
    }

    void Parameters::zero_grad() {
        w->zero_grad();
    }

    Matrix *Parameters::get_weight() {
        return w->get_weight();
    }

    Matrix *Parameters::get_grad() {
        return w->get_grad();
    }

    Matrix *Parameters::get_m() {
        return m;
    }

    Matrix *Parameters::get_v() {
        return v;
    }

    int Parameters::get_t() {
        return t;
    }

    void Parameters::inc_t() {
        t ++;
    }

    std::string Parameters::serialize() {
        Shape shape = w->getShape();
        DATATYPE *w_data = w->get_weight()->getData();
        DATATYPE *m_data = m->getData();
        DATATYPE *v_data = v->getData();
        int data_width = sizeof(DATATYPE);
        int data_size = shape.size() * data_width;
        

        int tot_size = 0;
        tot_size += sizeof(data_width);
        tot_size += sizeof(data_size);
        tot_size += sizeof(shape.rowCnt);
        tot_size += sizeof(shape.colCnt);
        tot_size += sizeof(t);
        tot_size += data_size; // w_data
        tot_size += data_size; // m_data
        tot_size += data_size; // v_data
        
        auto buffer = new char[tot_size];
        auto offset = 0;
        memcpy(buffer + offset, &data_width, sizeof(data_width));
        offset += sizeof(data_width);
        memcpy(buffer + offset, &data_size, sizeof(data_size));
        offset += sizeof(data_size);
        memcpy(buffer + offset, &shape.rowCnt, sizeof(shape.rowCnt));
        offset += sizeof(shape.rowCnt);
        memcpy(buffer + offset, &shape.colCnt, sizeof(shape.colCnt));
        offset += sizeof(shape.colCnt);
        memcpy(buffer + offset, &t, sizeof(t));
        offset += sizeof(t);
        memcpy(buffer + offset, w_data, data_size);
        offset += data_size;
        memcpy(buffer + offset, m_data, data_size);
        offset += data_size;
        memcpy(buffer + offset, v_data, data_size);
        std::string res((char *)buffer, tot_size);
        delete [] buffer;
        return res;
    }

    void Parameters::deserialize(char *buffer) {
        assert(w != nullptr);
        assert(w->get_weight() != nullptr);
        int data_width;
        int data_size;
        int rowCnt;
        int colCnt;
        auto offset = 0;
        memcpy(&data_width, buffer + offset, sizeof(data_width));
        if (data_width != sizeof(DATATYPE)) {
            std::cerr << "data width mismatch." << std::endl;
            abort();
        }
        offset += sizeof(data_width);
        memcpy(&data_size, buffer + offset, sizeof(data_size));
        offset += sizeof(data_size);
        memcpy(&rowCnt, buffer + offset, sizeof(rowCnt));
        offset += sizeof(rowCnt);
        memcpy(&colCnt, buffer + offset, sizeof(colCnt));
        offset += sizeof(colCnt);
        memcpy(&t, buffer + offset, sizeof(t));
        offset += sizeof(t);
        Shape shape(rowCnt, colCnt);
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wsign-compare"    
        assert(data_size == shape.size() * data_width);
        #pragma GCC diagnostic pop
        memcpy(w->get_weight()->getData(), buffer + offset, data_size);
        offset += data_size;
        memcpy(m->getData(), buffer + offset, data_size);
        offset += data_size;
        memcpy(v->getData(), buffer + offset, data_size);
    }

    bool Parameters::require_grad() {
        return w->is_require_grad();
    }
} // namespace autograd