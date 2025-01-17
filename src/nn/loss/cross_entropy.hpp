#include "math.h"
#include "matrix.hpp"
#include "loss.hpp"

class CrossEntropyLoss : public Loss
{
public:
    CrossEntropyLoss();
    float calculate_loss(Matrix predicted, Matrix expected);
    Matrix calculate_derivative(Matrix predicted, Matrix expected);
};

CrossEntropyLoss::CrossEntropyLoss() {}

float CrossEntropyLoss::calculate_loss(Matrix predicted, Matrix expected)
{
    assert(predicted.rows() == expected.rows());
    assert(predicted.cols() == expected.cols());

    // Calculate the loss for each element in the matrix
    // using the formula -y*log(y_hat) and sum the results
    Matrix result = Matrix(expected.rows(), expected.cols());
    for (int i = 0; i < predicted.rows(); i++)
    {
        for (int j = 0; j < predicted.cols(); j++)
        {
            result.set(i, j, expected.get(i, j) * log(predicted.get(i, j)));
        }
    }
    return -1.0 * result.sum();
}

Matrix CrossEntropyLoss::calculate_derivative(Matrix predicted, Matrix expected)
{
    return predicted - expected;
}
