#ifndef RELU_HPP
#define RELU_HPP

#include "matrix.hpp"
#include "elementwise_activation.hpp"

class ReLu: public ElementwiseActivation{
    public:
        ReLu();
        matrix_t activation(matrix_t value);
        Matrix calculate_derivative(Matrix output);
};

ReLu::ReLu(){}

matrix_t ReLu::activation(matrix_t value){
    if (value < 0){
        return 0;
    } else {
        return value;
    }
}

Matrix ReLu::calculate_derivative(Matrix output){
    Matrix result = Matrix(output.rows(), output.cols());
    for(int i=0; i<output.rows(); i++){
        for(int j=0; j<output.cols(); j++){
            if (output.get(i, j) < 0){
                result.set(i, j, 0);
            } else {
                result.set(i, j, 1);
            }
        }
    }
    return result;
}

#endif