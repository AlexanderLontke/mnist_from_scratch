#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <math.h>
#include "matrix.hpp"
#include "activation.hpp"

class Softmax : public Activation {
public:
    Softmax();

    Matrix _handle_forward(const Matrix &input) override;

    Matrix calculate_derivative(const Matrix &input) const override;
};

inline Softmax::Softmax() {
}

inline Matrix Softmax::_handle_forward(const Matrix &input) {
    Matrix result = Matrix(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        // subtract max per row for numerical stability
        matrix_t maxv = input.get(i, 0);
        for (int j = 1; j < input.cols(); j++) {
            if (input.get(i, j) > maxv) maxv = input.get(i, j);
        }
        matrix_t sum = 0;
        for (int j = 0; j < input.cols(); j++) {
            matrix_t e = exp(input.get(i, j) - maxv);
            result.set(i, j, e);
            sum += e;
        }
        for (int j = 0; j < input.cols(); j++) {
            result.set(i, j, result.get(i, j) / sum);
        }
    }
    return result;
}

inline Matrix Softmax::calculate_derivative(const Matrix &input) const {
    //iterate over i and j indices and calculate the derivative
    // of the softmax function using the kronicker delta
    Matrix result = Matrix(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            if (i == j) {
                result.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
            } else {
                result.set(i, j, -input.get(i, j) * input.get(j, i));
            }
        }
    }
    return result;
};

#endif
