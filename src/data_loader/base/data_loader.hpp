#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <tuple>
#include <vector>

#include "matrix.hpp"

template <typename T_class>
class DataLoader
{
public:
    DataLoader();
    virtual ~DataLoader();

    virtual std::tuple<Matrix, int> next() = 0;

    virtual bool empty() const = 0;
    virtual int size() const = 0;
    virtual void reset() { throw std::runtime_error("DataLoader: reset() not implemented"); }
};


template <typename T_constr>
DataLoader<T_constr>::DataLoader() = default;

template <typename T_de_constr>
DataLoader<T_de_constr>::~DataLoader() {}

#endif