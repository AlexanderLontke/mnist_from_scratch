// cpp
#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <numeric>
#include <vector>
#include <stdexcept>

#include "data_loader.hpp"
#include "loss.hpp"
#include "matrix.hpp"
#include "metrics/base/Metric.h"
#include "model.hpp"

template<typename T_dataloader>
class ModelTrainer {
public:
  ModelTrainer(Model &model, Loss &loss,
               DataLoader<T_dataloader> &train_dataloader,
               DataLoader<T_dataloader> &test_dataloader, int epochs,
               Metric &metric, int batch_size = 64);

  void train();

private:
  Model &_model;
  Loss &_loss;
  DataLoader<T_dataloader> &_train_dataloader;
  DataLoader<T_dataloader> &_test_dataloader;
  int _epochs;
  Metric &eval_metric;
  int _batch_size;
};

template<typename T_dataloader>
ModelTrainer<T_dataloader>::ModelTrainer(
  Model &model, Loss &loss, DataLoader<T_dataloader> &train_dataloader,
  DataLoader<T_dataloader> &test_dataloader, int epochs,
  Metric &metric, int batch_size)
  : _model(model), _loss(loss), _train_dataloader(train_dataloader),
    _test_dataloader(test_dataloader), eval_metric(metric), _batch_size(batch_size) {
  this->_epochs = epochs;
}

template<typename T_dataloader>
void ModelTrainer<T_dataloader>::train() {
  for (int epoch = 0; epoch < this->_epochs; epoch++) {
    this->_train_dataloader.reset();
    this->_model.training_mode();
    auto losses = std::vector<float>();

    while (!this->_train_dataloader.empty()) {
      // accumulate up to _batch_size samples
      std::vector<Matrix> batch_inputs_vec;
      std::vector<int> batch_labels_vec;
      batch_inputs_vec.reserve(this->_batch_size);
      batch_labels_vec.reserve(this->_batch_size);

      while (!this->_train_dataloader.empty() && (int)batch_inputs_vec.size() < this->_batch_size) {
        auto [input, label] = this->_train_dataloader.next();
        batch_inputs_vec.push_back(input);
        batch_labels_vec.push_back(label);
      }

      if (batch_inputs_vec.empty()) break;

      // Build batch Matrix (rows = batch_size_collected, cols = input_dim)
      int collected = static_cast<int>(batch_inputs_vec.size());
      int input_cols = batch_inputs_vec[0].cols();
      Matrix batch_inputs(collected, input_cols);
      for (int i = 0; i < collected; ++i) {
        // assumes each input is 1 x input_cols
        for (int c = 0; c < input_cols; ++c) {
          batch_inputs.set(i, c, batch_inputs_vec[i].get(0, c));
        }
      }

      // Forward
      Matrix outputs = this->_model.forward(batch_inputs);

      // Build one-hot labels (batch x num_classes)
      int num_classes = outputs.cols();
      Matrix one_hot_labels(collected, num_classes);
      one_hot_labels.fill(0.0f);
      for (int i = 0; i < collected; ++i) {
        Matrix row = Matrix::one_hot_encode(batch_labels_vec[i], num_classes);
        for (int j = 0; j < num_classes; ++j) {
          one_hot_labels.set(i, j, row.get(0, j));
        }
      }

      // Loss, backward, update
      losses.push_back(this->_loss.calculate_loss(outputs, one_hot_labels));
      this->_model.backpropagation(one_hot_labels, outputs, this->_loss);
      this->_model.update_weights();
    };

    float epoch_loss =
        std::accumulate(losses.begin(), losses.end(), 0.0f) / (losses.empty() ? 1.0f : (float)losses.size());
    std::cout << "Epoch " << epoch + 1 << "/" << this->_epochs << " completed (Loss: "
        << epoch_loss << ")" << std::endl;

    this->_model.evaluation_mode();

    this->_test_dataloader.reset();
    std::vector<Matrix> outputs;
    std::vector<int> labels;
    while (!this->_test_dataloader.empty()) {
      auto [inputs, label] = this->_test_dataloader.next();
      outputs.push_back(this->_model.forward(inputs));
      labels.push_back(label);
    }
    const float accuracy = eval_metric.calculate(outputs, labels);
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;
  }
};

#endif
