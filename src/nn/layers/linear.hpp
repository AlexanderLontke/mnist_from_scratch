// cpp
#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "matrix.hpp"
#include "layer.hpp"

class Linear final : public Layer
{
public:
    Linear(int input_size, int output_size);
    void update_weights(float learning_rate);
    int input_size = 0;
    int output_size = 0;
    Matrix weights;
    Matrix bias;

private:
    Matrix _handle_forward(Matrix input) override;
};

inline Linear::Linear(const int input_size, const int output_size)
    : input_size(input_size), output_size(output_size)
{
    // weights: (input_size x output_size)
    // bias: (1 x output_size)
    this->weights = Matrix(input_size, output_size).random();
    this->bias = Matrix(1, output_size).random();
}

inline void Linear::update_weights(float learning_rate)
{
    // Uses cached input (_input_cache) and stored delta (Layer::_delta)
    // Math:
    //   outputs = X W + b
    //   given Δ = dL/d(outputs)  (shape: batch x output_size)
    //   weight gradient: dL/dW = X^T · Δ    (shape: input_size x output_size)
    //   bias gradient:   dL/db = mean_rows(Δ)  (shape: 1 x output_size)
    //
    // Update (gradient descent):
    //   W <- W - lr * (dL/dW) / batch
    //   b <- b - lr * mean_rows(Δ)
    if (this->_input_cache.cols() == 0) {
        throw std::invalid_argument("Linear::update_weights: Input matrix is empty");
    };

    Matrix grad_w = this->_input_cache.transpose() * this->_delta; //  Δ x X^T

    // apply averaged gradient
    this->weights += grad_w * (-learning_rate);

    // bias: apply averaged delta; relies on broadcasting of Δ when adding to (1 x output_size)
    this->bias += (this->_delta * (-learning_rate));
}

inline Matrix Linear::_handle_forward(const Matrix input)
{
    return (input * this->weights) + this->bias;
}

#endif
