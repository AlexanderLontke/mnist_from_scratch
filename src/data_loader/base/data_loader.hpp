#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <iostream>
#include <tuple>
#include <vector>

template <typename T>
class DataLoader
{
public:
    DataLoader();
    virtual ~DataLoader();

    virtual std::tuple<std::vector<T>, std::vector<int> > next() = 0;

    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
};

DataLoader<T>::DataLoader()
{
}

DataLoader<T>::~DataLoader()
{
}

#endif