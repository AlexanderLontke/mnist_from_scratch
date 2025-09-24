// cpp
#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "matrix.hpp"
#include "layer.hpp"
#include <stdexcept>

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
    Matrix _handle_forward(const Matrix &input) override;
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
    if (this->_input_cache.cols() == 0) {
        throw std::invalid_argument("Linear::update_weights: Input matrix is empty");
    };

    const int batch_size = this->_input_cache.rows();
    if (batch_size <= 0) return;

    // weight gradient: X^T · Δ  (input_size x output_size)
    Matrix grad_w = this->_input_cache.transpose() * this->_delta;

    // average over batch
    float inv_batch = 1.0f / static_cast<float>(batch_size);
    grad_w = grad_w * inv_batch;

    // update weights (gradient descent)
    this->weights += grad_w * (-learning_rate);

    // bias gradient: mean over rows of Δ  -> shape (1 x output_size)
    Matrix grad_b(1, this->_delta.cols());
    for (int c = 0; c < this->_delta.cols(); ++c) {
        float s = 0.0f;
        for (int r = 0; r < this->_delta.rows(); ++r) {
            s += this->_delta.get(r, c);
        }
        grad_b.set(0, c, s * inv_batch);
    }

    // apply bias update
    this->bias += (grad_b * (-learning_rate));
}

inline Matrix Linear::_handle_forward(const Matrix &input)
{
    // compute linear output
    Matrix out = (input * this->weights);

    // if bias is a single row and out has multiple rows, broadcast bias
    if (this->bias.rows() == 1 && out.rows() > 1) {
        Matrix bcast(out.rows(), this->bias.cols());
        for (int r = 0; r < bcast.rows(); ++r) {
            for (int c = 0; c < bcast.cols(); ++c) {
                bcast.set(r, c, this->bias.get(0, c));
            }
        }
        return out + bcast;
    }

    // otherwise sizes should match (or bias already matches)
    return out + this->bias;
}

#endif
