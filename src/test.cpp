//
// Created by Lontke Alexander on 24.09.2025.
//

// src/main.cpp
#include <memory>
#include "feed_forward_neural_network.hpp"
#include "matrix.hpp"
#include "test_dataloader.hpp"
#include "metrics/MulticlassAccuracy.h"
#include "trainer.hpp"
#include "nn/loss/cross_entropy.hpp"

int main() {

    auto train_dataloader = std::make_unique<TestDataLoader>();
    auto test_dataloader = std::make_unique<TestDataLoader>();

    auto ffnn = std::make_unique<FeedForwardNeuralNetwork>(2, 2, 2, 0.01);
    auto multiclass_accuracy = std::make_unique<MulticlassAccuracy>();
    auto cross_entropy_loss = std::make_unique<CrossEntropyLoss>();

    auto model_trainer = std::make_unique<ModelTrainer<Matrix>>(
        *ffnn, *cross_entropy_loss, *train_dataloader, *test_dataloader, 200, *multiclass_accuracy
    );

    model_trainer->train();

    return 0;
}
