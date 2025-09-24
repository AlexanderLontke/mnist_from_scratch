#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "loss.hpp"
#include <cmath>
#include "matrix.hpp"

#include <cassert>

class CrossEntropyLoss final : public Loss {
public:
  CrossEntropyLoss();

  float calculate_loss(Matrix predicted, Matrix expected) override;

  Matrix calculate_derivative(Matrix predicted, Matrix expected) override;
};

inline CrossEntropyLoss::CrossEntropyLoss() = default;

inline float CrossEntropyLoss::calculate_loss(const Matrix predicted,
                                              const Matrix expected) {
  assert(predicted.rows() == expected.rows());
  assert(predicted.cols() == expected.cols());

  // Calculate the loss for each element in the matrix
  // using the formula -y*log(y_hat) and sum the results
  // epsilon for numerical stability
  const matrix_t epsilon = 1e-12;
  Matrix predicted_clipped = predicted;
  for (int i = 0; i < predicted.rows(); i++) {
    for (int j = 0; j < predicted.cols(); j++) {
      if (predicted_clipped.get(i, j) < epsilon) {
        predicted_clipped.set(i, j, epsilon);
      } else if (predicted_clipped.get(i, j) > 1 - epsilon) {
        predicted_clipped.set(i, j, 1 - epsilon);
      }
    }
  }
  Matrix result = Matrix(expected.rows(), expected.cols());
  for (int i = 0; i < predicted_clipped.rows(); i++) {
    for (int j = 0; j < predicted_clipped.cols(); j++) {
      result.set(i, j, expected.get(i, j) * log(predicted_clipped.get(i, j)));
    }
  }
  return -1.0 * result.sum();
}

inline Matrix CrossEntropyLoss::calculate_derivative(const Matrix predicted,
                                                     const Matrix expected) {
  return predicted - expected;
}
#endif // CROSS_ENTROPY_HPP
