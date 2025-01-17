#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <math.h>
#include "matrix.hpp"
#include "activation.hpp"

class Softmax: public Activation{
    public:
        Softmax();
        Matrix _handle_forward(Matrix input);
        Matrix calculate_derivative(Matrix input);
};

Softmax::Softmax(){}

Matrix Softmax::_handle_forward(Matrix input){
    // Todo: Make stable using the mean of the input

    Matrix result = Matrix(input.rows(), input.cols());
    for(int i=0; i<input.rows(); i++){
        // Calculate mean of the input
        matrix_t mean = 0.0F;
        for(int j=0; j<input.cols(); j++){
            mean += input.get(i, j);
        }
        mean /= input.cols();
        // Calculate the softmax function
        // subtract the mean from the input to 
        // avoid large exponents
        matrix_t sum = 0;
        for(int j=0; j<input.cols(); j++){
            sum += exp(input.get(i, j) - mean);
        }
        for(int j=0; j<input.cols(); j++){
            result.set(i, j, exp(input.get(i, j) - mean)/sum);
        }
    }
    return result;
}

Matrix Softmax::calculate_derivative(Matrix input){
    //iterate over i and j indices and calculate the derivative
    // of the softmax function using the kronicker delta
    Matrix result = Matrix(input.rows(), input.cols());
    for(int i=0; i<input.rows(); i++){
        for(int j=0; j<input.cols(); j++){
            if (i == j){
                result.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
            } else {
                result.set(i, j, -input.get(i, j) * input.get(j, i));
            }
        }
    }
    return result;
};

#endif