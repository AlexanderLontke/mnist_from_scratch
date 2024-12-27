#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "matrix.hpp"
#include "layer.hpp"

class Activation: public Layer{
    public:
        virtual matrix_t activation(matrix_t value) = 0;
    private:
        Matrix _handle_forward(Matrix input);
};

Matrix Activation::_handle_forward(Matrix input){
    Matrix result = Matrix(input.rows(), input.cols());
    for(int i=0; i<input.rows(); i++){
        for(int j=0; j<input.cols(); j++){
            result.set(i, j, this->activation(input.get(i, j)));
        }
    }
    return result;
}
#endif