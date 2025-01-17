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

class FeedForwardNeuralNetwork : public Model
{
public:
    FeedForwardNeuralNetwork(int input_size, int hidden_size, int output_size, float learning_rate);
    Matrix forward(Matrix input);
    void backpropagation(Matrix expected_output, Loss& loss);
    void update_weights(float learning_rate);
    // Define layers
    Linear* linear_1;
    ReLu* relu;
    Linear* linear_2;
    Softmax* softmax;
    float _learning_rate;
};

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(int input_size, int hidden_size, int output_size, float learning_rate)
{
    this->linear_1 = new Linear(input_size, hidden_size);
    this->relu = new ReLu();
    this->linear_2 = new Linear(hidden_size, output_size);
    this->softmax = new Softmax();
    this->_learning_rate = 0.01;
}

Matrix FeedForwardNeuralNetwork::forward(Matrix input)
{
    Matrix output = input;
    output = this->linear_1->forward(output);
    output = this->relu->forward(output);
    output = this->linear_2->forward(output);
    output = this->softmax->forward(output);
    return output;
}

void FeedForwardNeuralNetwork::backpropagation(Matrix expected_output, Loss& loss)
{
    // Calculate loss
    Matrix predicted_output = this->softmax->get_outputs();
    float loss_value = loss.calculate_loss(predicted_output, expected_output);
    std::cout << "Loss: " << loss_value << std::endl;

    // Backpropagate the error
    this->linear_2->_delta = loss.calculate_derivative(predicted_output, expected_output) * this->softmax->backward();
    this->linear_1->_delta = this->linear_2->_delta * this->linear_2->weights.transpose() * this->relu->backward();
}

void FeedForwardNeuralNetwork::update_weights(float learning_rate)
{
    this->linear_1->update_weights(learning_rate);
    this->linear_2->update_weights(learning_rate);
}

#endif