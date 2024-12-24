#include "matrix.hpp"
#include "activation.hpp"

class Sigmoid: public Activation{
    public:
        Sigmoid();
        matrix_t activation(matrix_t value);
};

Sigmoid::Sigmoid(){}

matrix_t Sigmoid::activation(matrix_t value){
    return 1/(1 + exp(-value));
}
