//
// Created by Lontke Alexander on 22.09.2025.
//

#include "MulticlassAccuracy.h"

float MulticlassAccuracy::calculate(const std::vector<Matrix> &outputs,
                                    const std::vector<int> &labels) {
  int correct = 0;
  for (int i = 0; i < outputs.size(); i++) {
    int predicted = 0;
    const Matrix& row = outputs.at(i);
    float max_prob = row.get(0, 0);
    for (int j = 1; j < row.cols(); j++) {
      if (row.get(0, j) > max_prob) {
        max_prob = row.get(0, j);
        predicted = j;
      }
    }
    if (predicted == labels.at(i)) {
      correct++;
    }
  }
  return static_cast<float>(correct) / outputs.size();
}