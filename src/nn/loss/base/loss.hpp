#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

class Loss
{
public:
    ~Loss();
    virtual float calculate_loss(Matrix predicted, Matrix expected) = 0;
    virtual Matrix calculate_derivative(Matrix predicted, Matrix expected) = 0;
};

Loss::~Loss() {}

#endif