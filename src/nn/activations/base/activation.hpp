// cpp
#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "matrix.hpp"
#include "layer.hpp"

class Activation: public Layer{
public:
    Activation() = default;
    Matrix backward();
    virtual Matrix calculate_derivative(const Matrix& input) const = 0;
    static bool is_activation_layer() { return true; };
};

inline Matrix Activation::backward(){
    return this->calculate_derivative(this->get_input_cache());
}
#endif
