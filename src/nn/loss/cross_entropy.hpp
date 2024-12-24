#include "layer.hpp"

class CrossEntropyLoss: public Layer{
public:
    CrossEntropyLoss();
    float forward(Matrix y_hat, Matrix y);
};

CrossEntropyLoss::CrossEntropyLoss(){}

float CrossEntropyLoss::forward(Matrix y_hat, Matrix y){
    Matrix result = Matrix(y.rows(), y.cols());
    for(int i=0; i<y.rows(); i++){
        for(int j=0; j<y.cols(); j++){
            result.set(i, j, y.get(i, j)*log(y_hat.get(i, j)));
        }
    }
    return -1.0 * result.sum();
}