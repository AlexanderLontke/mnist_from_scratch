#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include <math.h>
#include "matrix.hpp"
#include "elementwise_activation.hpp"

class Sigmoid: public ElementwiseActivation{
    public:
        Sigmoid();
        matrix_t activation(matrix_t value) override;
        Matrix calculate_derivative(const Matrix& input) const override;
};

inline Sigmoid::Sigmoid(){}

inline matrix_t Sigmoid::activation(matrix_t value){
    return 1/(1 + exp(-value));
}

inline Matrix Sigmoid::calculate_derivative(const Matrix &input) const {
    return input * (Matrix(input.rows(), input.cols(), 1) - input);
}
#endif