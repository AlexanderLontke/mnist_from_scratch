#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include "matrix.hpp"
#include "layer.hpp"
#include "loss.hpp"

class Model
{
public:
    ~Model();
    virtual Matrix forward(Matrix input) = 0;
    virtual void backpropagation(Matrix expected_output, Loss &loss) = 0;
    virtual void update_weights(float learning_rate) = 0;
};

Model::~Model() {};

#endif