#include "matrix.hpp"
#include "trainer.hpp"


// main function
int main() {
    // printing hello world
    Matrix m(2, 3, 1.0);
    (m * 2.0).print();

    Eye e(2, 2);
    e.print();

    // FeedForwardNeuralNetwork nn(32, 128, 10, "relu");
    return 0;
}