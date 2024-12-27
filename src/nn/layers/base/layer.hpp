#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

class Layer
{
public:
    Layer();
    ~Layer();
    Matrix forward(Matrix input);
    virtual void update_weights(Matrix update_gradients) = 0;

private:
    virtual Matrix _handle_forward(Matrix input) = 0;
    Matrix _outputs;
    Matrix _update_gradients;
};

Layer::Layer()
{
    this->_outputs = Matrix(0, 0);
    this->_update_gradients = Matrix(0, 0);
}

Layer::~Layer()
{
}

Matrix Layer::forward(Matrix input)
{
    this->_outputs = this->_handle_forward(input);
    return this->_outputs;
}

#endif