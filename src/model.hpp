#include <iostream>
#include "linear.hpp"
#include "matrix.hpp"
#include "layer.hpp"
#include "activation.hpp"

class NeuralNetwork{
    public:
        NeuralNetwork(int input_size, int hidden_size, int output_size, std::string activation);
        Matrix forward(Matrix input);
        void backward(Matrix grad_output, matrix_t learning_rate);
    private:
        std::vector<Layer> layers;
};

Matrix NeuralNetwork::forward(Matrix input){
    Matrix output = input;
    for(int i=0; i<this->layers.size(); i++){
        output = this->layers[i].forward(output);
    }
    return output;
}