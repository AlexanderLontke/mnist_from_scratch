#include <fstream>
#include "data_loader.hpp"
#include "matrix.hpp"

class MNISTDataLoader : public DataLoader<Matrix>
{
public:
    MNISTDataLoader(std::string images_file, std::string labels_file);
    std::tuple<std::vector<Matrix>, std::vector<int>> next();
    bool empty() const;
    size_t size() const;

private:
    std::vector<Matrix> images;
    std::vector<int> labels;
};

MNISTDataLoader::MNISTDataLoader(std::string images_file, std::string labels_file)
{
    std::ifstream images_stream(images_file, std::ios::binary);
    std::ifstream labels_stream(labels_file, std::ios::binary);

    // Read images
    

