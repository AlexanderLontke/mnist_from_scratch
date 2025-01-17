#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

class Layer
{
public:
    Layer();
    ~Layer();
    Matrix forward(Matrix input);
    Matrix get_outputs() { return this->_outputs; };
    Matrix get_delta() { return this->_delta; };
    
    // Hold delta values for backpropagation
    Matrix _delta;
protected:
    // Cache outputs for backpropagation
    Matrix _outputs;
    virtual Matrix _handle_forward(Matrix input) = 0;
};


Layer::Layer()
{
    this->_outputs = Matrix(0, 0);
    this->_delta = Matrix(0, 0);
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