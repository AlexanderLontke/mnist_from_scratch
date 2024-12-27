#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "matrix.hpp"
#include "layer.hpp"

class Linear : public Layer
{
public:
    Linear(int input_size, int output_size);
    void update_weights(Matrix update_gradients);
private:
    int input_size;
    int output_size;
    Matrix weights;
    Matrix bias;
    Matrix _handle_forward(Matrix input);
};

Linear::Linear(int input_size, int output_size)
{
    this->input_size = input_size;
    this->output_size = output_size;
    this->weights = Matrix(input_size, output_size).random();
    this->bias = Matrix(1, output_size).random();
}

Matrix Linear::_handle_forward(Matrix input)
{
    return (input * (this->weights)) + this->bias;
}

void Linear::update_weights(Matrix update_gradients)
{
    this->weights += update_gradients;
}

#endif