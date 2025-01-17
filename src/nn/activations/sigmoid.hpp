#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include <math.h>
#include "matrix.hpp"
#include "elementwise_activation.hpp"

class Sigmoid: public ElementwiseActivation{
    public:
        Sigmoid();
        matrix_t activation(matrix_t value);
        Matrix calculate_derivative(Matrix input);
};

Sigmoid::Sigmoid(){}

matrix_t Sigmoid::activation(matrix_t value){
    return 1/(1 + exp(-value));
}

Matrix Sigmoid::calculate_derivative(Matrix input){
    return input * (Matrix(input.rows(), input.cols(), 1) - input);
}
#endif