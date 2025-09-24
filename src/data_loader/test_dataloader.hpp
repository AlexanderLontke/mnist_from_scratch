//
// Created by Lontke Alexander on 24.09.2025.
//

#ifndef MNIST_FROM_SCRATCH_TEST_DATALOADER_H
#define MNIST_FROM_SCRATCH_TEST_DATALOADER_H

// Create dataloader that returns a fixed image and label for testing purposes
#include "data_loader.hpp"
#include "matrix.hpp"

class TestDataLoader final : public DataLoader<Matrix> {
public:
    TestDataLoader();
    std::tuple<Matrix, int> next() override;
    bool empty() const override;
    int size() const override;
    void reset() override;
private:
    int current_index;
    int total_size;
};

inline TestDataLoader::TestDataLoader() {
    this->current_index = 0;
    this->total_size = 1; // fixed size of 1 for testing
};


inline std::tuple<Matrix, int> TestDataLoader::next() {
    Matrix image = Matrix(1, 2, 0.25f); // fixed 1x2 image with values 0.5
    int label = 1; // fixed label
    this->current_index++;
    return std::make_tuple(image, label);
}

inline bool TestDataLoader::empty() const {
    return this->current_index >= this->total_size;
}
inline int TestDataLoader::size() const {
    return this->total_size;
}

inline void TestDataLoader::reset() {
    this->current_index = 0;
}
#endif //MNIST_FROM_SCRATCH_TEST_DATALOADER_H