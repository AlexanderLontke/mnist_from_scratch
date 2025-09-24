//
// Created by Lontke Alexander on 22.09.2025.
//

#ifndef MNIST_FROM_SCRATCH_METRIC_H
#define MNIST_FROM_SCRATCH_METRIC_H
#include <vector>

#include "matrix.hpp"

class Metric {
public:
  Metric();
  ~Metric();
  // Calculate metric based on model outputs and true labels
  virtual float calculate(const std::vector<Matrix> &outputs, const std::vector<int> &labels) = 0;
};

#endif // MNIST_FROM_SCRATCH_METRIC_H
