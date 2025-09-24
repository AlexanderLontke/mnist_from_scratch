#ifndef FEED_FORWARD_NEURAL_NETWORK_HPP
#define FEED_FORWARD_NEURAL_NETWORK_HPP

#include <iostream>
#include "linear.hpp"
#include "matrix.hpp"
#include "layer.hpp"
#include "model.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include "cross_entropy.hpp"

class FeedForwardNeuralNetwork : public Model {
public:
    FeedForwardNeuralNetwork(int input_size, int hidden_size, int output_size, float learning_rate);

    Matrix forward(Matrix input) override;

    void backpropagation(Matrix expected_output, Matrix predicted_output, Loss &loss) override;

    void update_weights() override;

    void training_mode() override;

    void evaluation_mode() override;

    // Define layers
    Linear * linear_0;
    ReLu *relu_0;
    Linear *linear_1;
    ReLu *relu_1;
    Linear *linear_2;
    Softmax *softmax;
    float _learning_rate;
};

inline FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(int input_size, int hidden_size, int output_size,
                                                          float learning_rate) {
    this->linear_0 = new Linear(input_size, hidden_size);
    this->relu_0 = new ReLu();
    this->linear_1 = new Linear(hidden_size, hidden_size);
    this->relu_1 = new ReLu();
    this->linear_2 = new Linear(hidden_size, output_size);
    this->softmax = new Softmax();
    this->_learning_rate = learning_rate;
}

inline Matrix FeedForwardNeuralNetwork::forward(Matrix input) {
    Matrix output = input;
    output = this->linear_0->forward(output);
    output = this->relu_0->forward(output);
    output = this->linear_1->forward(output);
    output = this->relu_1->forward(output);
    output = this->linear_2->forward(output);
    output = this->softmax->forward(output);
    return output;
}

inline void FeedForwardNeuralNetwork::backpropagation(Matrix expected_output, Matrix predicted_output, Loss &loss) {
    // Backpropagate the error
    this->linear_2->_delta = loss.calculate_derivative(predicted_output, expected_output);
    // Backpropagate to previous layer:
    // delta_hidden = (W^T · delta_output) ⊙ ReLU'(z_hidden)
    this->linear_1->_delta = (
        this->linear_2->_delta * this->linear_2->weights.transpose()
    ).dot_multiply(this->relu_1->backward());
    this->linear_0->_delta = (
        this->linear_1->_delta * this->linear_1->weights.transpose()
    ).dot_multiply(this->relu_0->backward());
}

inline void FeedForwardNeuralNetwork::update_weights() {
    this->linear_0->update_weights(this->_learning_rate);
    this->linear_1->update_weights(this->_learning_rate);
    this->linear_2->update_weights(this->_learning_rate);
}

inline void FeedForwardNeuralNetwork::training_mode() {
    this->linear_0->training_mode();
    this->relu_0->training_mode();
    this->linear_1->training_mode();
    this->relu_1->training_mode();
    this->linear_2->training_mode();
    this->softmax->training_mode();
};

inline void FeedForwardNeuralNetwork::evaluation_mode() {
    this->linear_0->evaluation_mode();
    this->relu_0->evaluation_mode();
    this->linear_1->evaluation_mode();
    this->relu_1->evaluation_mode();
    this->linear_2->evaluation_mode();
    this->softmax->evaluation_mode();
};

#endif
