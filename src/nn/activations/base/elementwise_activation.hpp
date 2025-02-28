#ifndef ELEMENTWISE_ACTIVATION_HPP
#define ELEMENTWISE_ACTIVATION_HPP

#include "matrix.hpp"
#include "layer.hpp"
#include "activation.hpp"

class ElementwiseActivation: public Activation{
    public:
        virtual matrix_t activation(matrix_t value) = 0;
        virtual Matrix calculate_derivative(Matrix input) = 0;
        Matrix update_weights(Matrix update_gradients);
        Matrix _handle_forward(Matrix input);
};

Matrix ElementwiseActivation::_handle_forward(Matrix input){
    Matrix result = Matrix(input.rows(), input.cols());
    for(int i=0; i<input.rows(); i++){
        for(int j=0; j<input.cols(); j++){
            result.set(i, j, this->activation(input.get(i, j)));
        }
    }
    return result;
}

Matrix ElementwiseActivation::update_weights(Matrix update_gradients){
    return this->calculate_derivative(update_gradients);
}
#endif