#include "matrix.hpp"
#include "activation.hpp"

class ReLu: public Activation{
    public:
        ReLu();
        matrix_t activation(matrix_t value);
};

ReLu::ReLu(){}

matrix_t ReLu::activation(matrix_t value){
    if (value < 0){
        return 0;
    } else {
        return value;
    }
}
