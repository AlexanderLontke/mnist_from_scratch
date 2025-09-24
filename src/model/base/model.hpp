#ifndef MODEL_HPP
#define MODEL_HPP

#include "matrix.hpp"
#include "loss.hpp"

class Model
{
public:
    virtual ~Model() = default;
    virtual Matrix forward(const Matrix &input) = 0;
    virtual void backpropagation(const Matrix &expected_output, const Matrix &predicted_output, Loss &loss) = 0;
    virtual void update_weights() = 0;
    virtual void training_mode() = 0;
    virtual void evaluation_mode() = 0;
};

#endif
