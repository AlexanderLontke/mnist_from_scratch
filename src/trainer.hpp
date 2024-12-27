#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "matrix.hpp"
#include "model.hpp"
#include "loss.hpp"
#include "data_loader.hpp"

class ModelTrainer
{
public:
    ModelTrainer(Model &model, Loss &loss, DataLoader& train_dataloader, DataLoader& test_dataloader, int epochs, float learning_rate);
    void train(Matrix input, Matrix expected_output);
    void test(Matrix input, Matrix expected_output);

private:
    Model &_model;
    Loss &_loss;
    DataLoader &_train_dataloader;
    DataLoader &_test_dataloader;
    int _epochs;
    float _learning_rate;
};

ModelTrainer::ModelTrainer(Model &model, Loss &loss, DataLoader& train_dataloader, DataLoader& test_dataloader, int epochs, float learning_rate)
    : _model(model), _loss(loss), _train_dataloader(train_dataloader), _test_dataloader(test_dataloader) {
    this->_epochs = epochs;
    this->_learning_rate = learning_rate;
};

#endif