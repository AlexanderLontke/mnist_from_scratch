#include "matrix.hpp"

#include <cassert>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <iostream>

Matrix::Matrix(int rows, int cols, matrix_t val)
  : _rows(rows), _cols(cols), _data(rows* cols, val)
{}

Matrix& Matrix::init(int rows, int cols, matrix_t val){
    this->_rows = rows;
    this->_cols = cols;
    this->_data = std::vector<matrix_t>(rows*cols, val);
    return *this;
}

matrix_t Matrix::get(int row, int col) const{
    return this->_data[row*this->_cols +col];
}

void Matrix::set(int row, int col, matrix_t value){
    this->_data[row*this->_cols + col] = value;
}

Matrix Matrix::operator+(const Matrix& other) const {
    // Dimensions must match
    if (! (this->_rows == other.rows() && this->_cols == other.cols())) {
        const std::string msg = "Matrix addition: Dimension mismatch: (" + std::to_string(this->_rows) + ", " + std::to_string(this->_cols) + ") + (" + std::to_string(other.rows()) + ", " + std::to_string(other.cols()) + ")\n";
        throw std::runtime_error(msg);
    }
    
    // Define results matrix
    auto result = Matrix(this->_rows, this->_cols);

    // Apply element-wise addition
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) + other.get(i, j));
        }
    }
    // Return result
    return result;
}

Matrix Matrix::operator+(const matrix_t& value) const {
    // Define results matrix
    Matrix result = Matrix(this->_rows, this->_cols);

    // Apply element-wise addition
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) + value);
        }
    }
    // Return result
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    // Dimensions must match
    if (! (this->_rows == other.rows() && this->_cols == other.cols())) {
        const std::string msg = "Matrix subtraction: Dimension mismatch: (" + std::to_string(this->_rows) + ", " + std::to_string(this->_cols) + ") - (" + std::to_string(other.rows()) + ", " + std::to_string(other.cols()) + ")\n";
        throw std::runtime_error(msg);
    }

    // Define results matrix
    Matrix result = Matrix(this->_rows, this->_cols);

    // Apply element-wise subtraction
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) - other.get(i, j));
        }
    }

    // Return result
    return result;
}

Matrix Matrix::operator-(const matrix_t& value) const {
    // Define results matrix
    Matrix result = Matrix(this->_rows, this->_cols);

    // Apply element-wise subtraction
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) - value);
        }
    }

    // Return result
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (this->_cols != other.rows()){
        const std::string msg = "Matrix multiplication: Dimension mismatch: (" + std::to_string(this->_rows) + ", " + std::to_string(this->_cols) + ") * (" + std::to_string(other.rows()) + ", " + std::to_string(other.cols()) + ")\n";
        throw std::runtime_error(msg);
    }
    Matrix results = Matrix(this->_rows, other.cols());
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<other.cols(); j++){
            matrix_t tmp_sum = 0;
            for(int k=0; k<this->_cols; k++){
                tmp_sum += this->get(i, k) * other.get(k, j);
            }
            results.set(i, j, tmp_sum);
        }
    }
    return results;
}

Matrix Matrix::operator*(const matrix_t& value) const {
    Matrix result = Matrix(this->_rows, this->_cols);
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) * value);
        }
    }
    return result;
}

Matrix Matrix::dot_multiply(const Matrix& other) const {
    if (! (this->_rows == other.rows() && this->_cols == other.cols())) {
        const std::string msg = "Matrix dot multiplication: Dimension mismatch: (" + std::to_string(this->_rows) + ", " + std::to_string(this->_cols) + ") .* (" + std::to_string(other.rows()) + ", " + std::to_string(other.cols()) + ")\n";
        throw std::runtime_error(msg);
    }
    auto result = Matrix(this->_rows, this->_cols);
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) * other.get(i, j));
        }
    }

    return result;
}


Matrix Matrix::transpose() const {
    auto result = Matrix(this->_cols, this->_rows);
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(j, i, this->get(i, j));
        }
    }
    return result;
}
Matrix Matrix::one_hot_encode(const int label, const int num_classes) {
    auto return_matrix =  Matrix(1, num_classes);
    return_matrix.fill(0);
    return_matrix.set(0, label, 1);
    return return_matrix;
}

Matrix Matrix::random(const matrix_t low, const matrix_t high){
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            this->set(i, j, low + (matrix_t)((float)arc4random()/(float)RAND_MAX)*(high-low));
        }
    }
    return *this;
}

Matrix Matrix::fill(matrix_t value){
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            this->set(i, j, value);
        }
    }
    return *this;
}

std::vector<matrix_t> Matrix::data(){
    return this->_data;
}

std::vector<matrix_t> Matrix::data() const {
    return this->_data;
}

int Matrix::rows() const{
    return this->_rows;
}

int Matrix::cols() const{
    return this->_cols;
}

Matrix Matrix::operator+=(const Matrix& other){
    *this = *this + other;
    return *this;
}

Matrix Matrix::operator*=(const matrix_t& value){
    *this = *this * value;
    return *this;
}

Matrix Matrix::operator*=(const Matrix& other){
    *this = *this * other;
    return *this;
}

matrix_t Matrix::sum() const{
    matrix_t sum = 0;
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            sum += this->get(i, j);
        }
    }
    return sum;
}

void Matrix::print() const {
    const std::string format = "%.6f";
    std::ostringstream oss;
    oss << "[\n";
    for (int r = 0; r < _rows; ++r) {
        oss << "  ";
        for (int c = 0; c < _cols; ++c) {
            if (c != 0) oss << ", ";
            oss << std::fixed << std::setprecision(6) << this->get(r, c);
        }
        oss << "\n";
    }
    oss << "]\n";
    std::cout << oss.str();
}

Eye::Eye(int rows, int cols)
  : Matrix(rows, cols)
{
    for(int i=0; i<rows; i++){
        this->set(i, i, 1);
    }
}