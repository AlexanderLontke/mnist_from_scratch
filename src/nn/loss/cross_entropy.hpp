// cpp
#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "loss.hpp"
#include <cmath>
#include "matrix.hpp"

#include <cassert>

class CrossEntropyLoss final : public Loss {
public:
  CrossEntropyLoss();

  float calculate_loss(const Matrix &predicted, const Matrix &expected) override;

  Matrix calculate_derivative(const Matrix &predicted, const Matrix &expected) override;
};

inline CrossEntropyLoss::CrossEntropyLoss() = default;

inline float CrossEntropyLoss::calculate_loss(const Matrix &predicted,
                                              const Matrix &expected) {
  assert(predicted.rows() == expected.rows());
  assert(predicted.cols() == expected.cols());

  if (predicted.rows() == 0) return 0.0f;

  const matrix_t epsilon = 1e-12f;
  Matrix predicted_clipped = predicted;
  for (int i = 0; i < predicted.rows(); i++) {
    for (int j = 0; j < predicted.cols(); j++) {
      if (predicted_clipped.get(i, j) < epsilon) {
        predicted_clipped.set(i, j, epsilon);
      } else if (predicted_clipped.get(i, j) > 1.0f - epsilon) {
        predicted_clipped.set(i, j, 1.0f - epsilon);
      }
    }
  }

  Matrix result = Matrix(expected.rows(), expected.cols());
  for (int i = 0; i < predicted_clipped.rows(); i++) {
    for (int j = 0; j < predicted_clipped.cols(); j++) {
      result.set(i, j, expected.get(i, j) * std::log(predicted_clipped.get(i, j)));
    }
  }

  float batch_sum = -1.0f * result.sum();
  return batch_sum / static_cast<float>(predicted.rows()); // average per sample
}

inline Matrix CrossEntropyLoss::calculate_derivative(const Matrix &predicted,
                                                     const Matrix &expected) {
  // For mean loss, derivative is (predicted - expected) / N
  Matrix diff = predicted - expected;
  const int n = predicted.rows();
  if (n > 1) {
    diff = diff * (1.0f / static_cast<float>(n));
  }
  return diff;
}
#endif // CROSS_ENTROPY_HPP
