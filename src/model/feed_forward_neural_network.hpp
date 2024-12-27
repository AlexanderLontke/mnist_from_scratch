#ifndef FEED_FORWARD_NEURAL_NETWORK_HPP
#define FEED_FORWARD_NEURAL_NETWORK_HPP

#include <iostream>
#include "linear.hpp"
#include "matrix.hpp"
#include "layer.hpp"
#include "model.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "cross_entropy.hpp"

class FeedForwardNeuralNetwork : public Model
{
public:
    FeedForwardNeuralNetwork(int input_size, int hidden_size, int output_size, std::string activation);
    Matrix forward(Matrix input);
    void backpropagation(Matrix expected_output, Loss& loss);

private:
    std::vector<Layer *> layers;
    float _learning_rate;
};

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(int input_size, int hidden_size, int output_size, std::string activation)
{
    this->layers.push_back(new Linear(input_size, hidden_size));
    this->layers.push_back(new ReLu());
    this->layers.push_back(new Linear(hidden_size, output_size));
    this->layers.push_back(new Sigmoid());
}

Matrix FeedForwardNeuralNetwork::forward(Matrix input)
{
    Matrix output = input;
    for (int i = 0; i < this->layers.size(); i++)
    {
        output = this->layers[i]->forward(output);
    }
    return output;
}
void FeedForwardNeuralNetwork::backpropagation(Matrix expected_output, Loss& loss)
{
    
};


#endif