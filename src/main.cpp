// src/main.cpp
#include <memory>
#include "data_loader/mnist_dataloader.hpp"
#include "feed_forward_neural_network.hpp"
#include "matrix.hpp"
#include "metrics/MulticlassAccuracy.h"
#include "trainer.hpp"
#include "nn/loss/cross_entropy.hpp"

int main() {
    std::string base_path =
            "/Users/lontkealexander/OneDrive - The Boston Consulting Group, Inc/Documents/GitHub/mnist_from_scratch/";

    auto train_dataloader = std::make_unique<MNISTDataLoader>(
        base_path + "data/train-images.idx3-ubyte",
        base_path + "data/train-labels.idx1-ubyte");

    auto test_dataloader = std::make_unique<MNISTDataLoader>(
        base_path + "data/t10k-images.idx3-ubyte",
        base_path + "data/t10k-labels.idx1-ubyte");

    auto ffnn = std::make_unique<FeedForwardNeuralNetwork>(784, 128, 10, 0.01);
    auto multiclass_accuracy = std::make_unique<MulticlassAccuracy>();
    auto cross_entropy_loss = std::make_unique<CrossEntropyLoss>();

    auto model_trainer = std::make_unique<ModelTrainer<Matrix>>(
        *ffnn, *cross_entropy_loss, *train_dataloader, *test_dataloader, 500, *multiclass_accuracy
    );

    model_trainer->train();

    return 0;
}
