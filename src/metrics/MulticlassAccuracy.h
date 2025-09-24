//
// Created by Lontke Alexander on 22.09.2025.
//

#ifndef MNIST_FROM_SCRATCH_MULTICLASS_ACCURACY_H
#define MNIST_FROM_SCRATCH_MULTICLASS_ACCURACY_H
#include "base/Metric.h"
#include "matrix.hpp"

class MulticlassAccuracy final : public Metric {
public:
  float calculate(const std::vector<Matrix> &outputs,
                  const std::vector<int> &labels) override;
};

#endif // MNIST_FROM_SCRATCH_MULTICLASS_ACCURACY_H
