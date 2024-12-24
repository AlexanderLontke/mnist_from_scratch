#include "matrix.hpp"
#include "layer.hpp"

class Linear: public Layer{
    public:
        Linear(int input_size, int output_size);
        Matrix forward(Matrix input);
        Matrix backward(Matrix grad_output);
        void update_weights(Matrix grad_output, matrix_t learning_rate);
    private:
        int input_size;
        int output_size;
        Matrix weights;
        Matrix bias;
};

Linear::Linear(int input_size, int output_size){
    this->input_size = input_size;
    this->output_size = output_size;
    this->weights = Matrix(input_size, output_size).random();
    this->bias = Matrix(1, output_size).random();
}

Matrix Linear::forward(Matrix input){
    return (input*(this->weights)) + this->bias;
}

Matrix Linear::backward(Matrix grad_output){
    return grad_output*(this->weights.transpose());
}
