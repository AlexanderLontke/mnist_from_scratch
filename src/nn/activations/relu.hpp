#ifndef RELU_HPP
#define RELU_HPP

#include "matrix.hpp"
#include "elementwise_activation.hpp"

class ReLu : public ElementwiseActivation {
public:
    ReLu() = default;

    matrix_t activation(matrix_t value) override;

    [[nodiscard]] Matrix calculate_derivative(const Matrix& input) const override;
};

inline matrix_t ReLu::activation(const matrix_t value) {
    if (value < 0) {
        return 0;
    }
    return value;
}

inline Matrix ReLu::calculate_derivative(const Matrix &input) const {
    Matrix result = Matrix(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            if (input.get(i, j) < 0) {
                result.set(i, j, 0);
            } else {
                result.set(i, j, 1);
            }
        }
    }
    return result;
}

#endif
