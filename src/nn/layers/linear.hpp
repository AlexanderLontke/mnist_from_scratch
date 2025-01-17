#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "matrix.hpp"
#include "layer.hpp"

class Linear : public Layer
{
public:
    Linear(int input_size, int output_size);
    Matrix forward(Matrix input);
    void update_weights(float learning_rate);
    int input_size;
    int output_size;
    Matrix weights;
    Matrix bias;
private:    
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

void Linear::update_weights(float learning_rate)
{
    this->weights += this->_delta * this->_outputs.transpose() * -learning_rate;
    this->bias += this->_delta * -learning_rate;
}

#endif