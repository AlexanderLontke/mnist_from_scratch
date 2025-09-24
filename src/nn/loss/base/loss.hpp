#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

class Loss {
public:
    virtual ~Loss();

    virtual float calculate_loss(const Matrix &predicted, const Matrix &expected) = 0;

    virtual Matrix calculate_derivative(const Matrix &predicted, const Matrix &expected) = 0;
};

Loss::~Loss() {
}

#endif
