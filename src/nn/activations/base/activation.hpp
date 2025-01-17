#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "matrix.hpp"
#include "layer.hpp"

class Activation: public Layer{
    public:
        Matrix forward(Matrix input);
        Matrix backward();
        virtual Matrix calculate_derivative(Matrix input) = 0;
        virtual Matrix _handle_forward(Matrix input) = 0;
        bool is_activation_layer() { return true; };
};

Matrix Activation::backward(){
    return this->calculate_derivative(this->get_outputs());
}
#endif