#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include "matrix.hpp"

class Model
{
public:
    ~Model();
    virtual Matrix forward(Matrix input) = 0;
    virtual void backpropagation(Matrix expected_output) = 0; 
};

Model::~Model(){};
#endif