#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <iostream>
#include <tuple>
#include <vector>

class DataLoader
{
public:
    DataLoader();
    virtual ~DataLoader();

    virtual std::tuple<std::vector<float>, std::vector<int> > next() = 0;

    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
};

DataLoader::DataLoader()
{
}

DataLoader::~DataLoader()
{
}

#endif