#include "matrix.hpp"
#include "trainer.hpp"
#include "data_loader/mnist_dataloader.hpp"

// main function
int main() {
    std::string base_path = "/Users/lontkealexander/OneDrive - The Boston Consulting Group, Inc/Documents/GitHub/mnist_from_scratch/";
    DataLoader<Matrix>* dataloader = new MNISTDataLoader(
        base_path + "/data/train-images.idx3-ubyte",
        base_path + "data/train-labels.idx1-ubyte");
    std::tuple<Matrix, int> sample = dataloader->next();
    std::cout << "Input Matrix: " << std::endl;
    std::get<0>(sample).print();

    std::cout << "Label: " << std::get<1>(sample) << std::endl;
    return 0;
}