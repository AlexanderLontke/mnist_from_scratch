#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <numeric>

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
               Metric &metric);

  void train();

private:
  Model &_model;
  Loss &_loss;
  DataLoader<T_dataloader> &_train_dataloader;
  // Test dataloader is optional
  DataLoader<T_dataloader> &_test_dataloader = nullptr;
  int _epochs;
  Metric &eval_metric;
};

template<typename T_dataloader>
ModelTrainer<T_dataloader>::ModelTrainer(
  Model &model, Loss &loss, DataLoader<T_dataloader> &train_dataloader,
  DataLoader<T_dataloader> &test_dataloader, int epochs,
  Metric &metric)
  : _model(model), _loss(loss), _train_dataloader(train_dataloader),
    _test_dataloader(test_dataloader), eval_metric(metric) {
  this->_epochs = epochs;
}

template<typename T_dataloader>
void ModelTrainer<T_dataloader>::train() {
  // Training loop
  for (int epoch = 0; epoch < this->_epochs; epoch++) {
    // Reset dataloader at beginning of epoch
    this->_train_dataloader.reset();
    // Set model to training mode
    this->_model.training_mode();
    auto losses = std::vector<float>();
    // Fetch batch from train dataloader
    while (!this->_train_dataloader.empty()) {
      auto [inputs, label] = this->_train_dataloader.next();
      // Forward pass
      Matrix outputs = this->_model.forward(inputs);
      // Convert labels to one-hot encoding
      Matrix one_hot_labels = Matrix::one_hot_encode(label, outputs.cols());
      // Compute loss and store it
      losses.push_back(this->_loss.calculate_loss(outputs, one_hot_labels));
      // Backward pass
      this->_model.backpropagation(one_hot_labels, outputs, this->_loss);
      // Update weights
      this->_model.update_weights();
    };
    float epoch_loss =
        std::accumulate(losses.begin(), losses.end(), 0.0f) / losses.size();
    std::cout << "Epoch " << epoch + 1 << "/" << this->_epochs << " completed (Loss: "
        << epoch_loss << ")" << std::endl;

    // Set model to evaluation mode
    this->_model.evaluation_mode();

    // Reset test dataloader
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
