// cpp
#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

class Layer {
public:
    Layer();

    virtual ~Layer();

    Matrix forward(const Matrix &input);

    void evaluation_mode() { this->in_evaluation_mode = true; };
    void training_mode() { this->in_evaluation_mode = false; };
    Matrix get_input_cache() const { return this->_input_cache; };

    // Hold delta values for backpropagation
    Matrix _delta;

protected:
    virtual Matrix _handle_forward(const Matrix &input) = 0;
    bool in_evaluation_mode = false;
    static bool is_activation_layer() { return false; };
    // Cache input for backpropagation
    Matrix _input_cache;
};


inline Layer::Layer() {
    this->_input_cache = Matrix(0, 0);
    this->_delta = Matrix(0, 0);
}

inline Layer::~Layer()
= default;

inline Matrix Layer::forward(const Matrix &input) {
    // Cache input for use in update_weights
    if (!this->in_evaluation_mode) {
        this->_input_cache = input;
    }
    // outputs
    return this->_handle_forward(input);
}

#endif
