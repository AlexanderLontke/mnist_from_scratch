#ifndef DATASET_HPP
#define DATASET_HPP

#include "matrix.hpp"

class Dataset
{
public:
    virtual int count() const = 0;
    virtual Matrix get_input(int index) const = 0;
    virtual Matrix get_output(int index) const = 0;
};
#endif