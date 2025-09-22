#ifndef MNIST_DATALOADER_HPP
#define MNIST_DATALOADER_HPP
#include <fstream>
#include "base/data_loader.hpp"
#include "matrix.hpp"


inline int reverseInt (const int i)
{
    const unsigned char c1 = i & 255;
    const unsigned char c2 = (i >> 8) & 255;
    const unsigned char c3 = (i >> 16) & 255;
    const unsigned char c4 = (i >> 24) & 255;

    return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + c4;
}

class MNISTDataLoader final : public DataLoader<Matrix>
{
public:
    MNISTDataLoader(const std::string& images_file, const std::string& labels_file);
    std::tuple<Matrix, int> next();
    bool empty() const;
    int size() const;
    const std::vector<Matrix>& get_images() const;

    const std::vector<int>& get_labels() const;

    /** the following constants are defined as per the values described at http://yann.lecun.com/exdb/mnist/ **/
    int LABEL_MAGIC = 2049;
    int IMAGE_MAGIC = 2051;
    int ROWS = 28;
    int COLUMNS = 28;

private:
    std::vector<Matrix> images;
    std::vector<int> labels;
    int size_ = 0;
    int current_index = 0;
};

inline MNISTDataLoader::MNISTDataLoader(const std::string& images_file, const std::string& labels_file) {
    std::ifstream images_stream(images_file, std::ios::binary);
    std::ifstream labels_stream(labels_file, std::ios::binary);

    // Read images
    if (images_stream.is_open()) {
        int magic_number = 0;
        images_stream.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
        magic_number = reverseInt(
            magic_number
        );
        if (magic_number != IMAGE_MAGIC) {
            std::string base_error_msg = "MNISTDataLoader: Invalid image file. ";
            throw std::runtime_error(base_error_msg + "Expected magic number" + std::to_string(IMAGE_MAGIC) + " but got " + std::to_string(magic_number));
        }
        int number_of_images = 0;
        images_stream.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
        number_of_images = reverseInt(
            number_of_images
        );
        int n_rows = 0;
        images_stream.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
        n_rows = reverseInt(
            n_rows
        );
        int n_columns = 0;
        images_stream.read(reinterpret_cast<char *>(&n_columns), sizeof(n_columns));
        n_columns = reverseInt(
            n_columns
        );
        this->size_ = number_of_images;
        std::cout << "MNISTDataLoader: Loading " << number_of_images << " images of size " << n_rows << "x" << n_columns << std::endl;

        for (int i = 0; i < number_of_images; ++i) {
            auto current_image = Matrix(ROWS, COLUMNS, 1);
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_columns; ++c) {
                    unsigned char temp = 0;
                    images_stream.read(reinterpret_cast<char *>(&temp), sizeof(temp));
                    current_image.set(
                        r,
                        c,
                            temp
                        );
                }
            }
            // add image to images vector
            this->images.push_back(current_image);
        }
    } else {
        throw std::runtime_error("MNISTDataLoader: Unable to open image file");
    }

    // Read labels
    if (labels_stream.is_open()) {
        int magic_number = 0;
        labels_stream.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
        magic_number = reverseInt(
            magic_number
        );
        if (magic_number != LABEL_MAGIC) {
            throw std::runtime_error("MNISTDataLoader: Invalid label file");
        }
        int number_of_labels = 0;
        labels_stream.read(reinterpret_cast<char *>(&number_of_labels), sizeof(number_of_labels));
        number_of_labels = reverseInt(
            number_of_labels
        );
        if (number_of_labels != this->size_) {
            throw std::runtime_error("MNISTDataLoader: Number of labels does not match number of images");
        }
        for (int i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            labels_stream.read(reinterpret_cast<char *>(&temp), sizeof(temp)).get();
            this->labels.push_back(static_cast<int>(temp));
        }
    } else {
        throw std::runtime_error("MNISTDataLoader: Unable to open label file");
    }
}

inline int MNISTDataLoader::size() const {
    return this->size_;
};

inline bool MNISTDataLoader::empty() const {
    return this->current_index >= this->size_;
};

inline std::tuple<Matrix, int> MNISTDataLoader::next() {
    if (this->empty()) {
        throw std::out_of_range("MNISTDataLoader: No more data to load");
    }
    auto image = this->images[this->current_index];
    auto label = this->labels[this->current_index];
    this->current_index++;
    return std::make_tuple(image, label);
}



#endif // MNIST_DATALOADER_HPP
