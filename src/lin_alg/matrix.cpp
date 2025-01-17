#include "matrix.hpp"
#include <stdlib.h>
#include <math.h>
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

Matrix Matrix::operator+(const Matrix& other){
    // Dimensions must match
    assert(this->_rows == other.rows() && this->_cols == other.cols());
    
    // Define results matrix
    Matrix result = Matrix(this->_rows, this->_cols);

    // Apply element-wise addition
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) + other.get(i, j));
        }
    }
    // Return result
    return result;
}

Matrix Matrix::operator+(const matrix_t& value){
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

Matrix Matrix::operator-(const Matrix& other){
    // Dimensions must match
    assert(this->_rows == other.rows() && this->_cols == other.cols());
    
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

Matrix Matrix::operator-(const matrix_t& value){    
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

Matrix Matrix::operator*(const Matrix& other){
    assert (this->_cols == other.rows());
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

Matrix Matrix::operator*(const matrix_t& value){
    Matrix result = Matrix(this->_rows, this->_cols);
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) * value);
        }
    }
    return result;
}

Matrix Matrix::dot_multiply(const Matrix& other){
    assert(this->_rows == other.rows() && this->_cols == other.cols());
    Matrix result = Matrix(this->_rows, this->_cols);
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(i, j, this->get(i, j) * other.get(i, j));
        }
    }
    return result;
}


Matrix Matrix::transpose(){
    Matrix result = Matrix(this->_cols, this->_rows);
    for(int i=0; i<this->_rows; i++){
        for(int j=0; j<this->_cols; j++){
            result.set(j, i, this->get(i, j));
        }
    }
    return result;
}

Matrix Matrix::random(matrix_t low, matrix_t high){
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

const std::vector<matrix_t> Matrix::data() const{
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