#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

typedef float matrix_t;

class Matrix
{
public:
    // constructor
    Matrix(int rows = 0, int cols = 0, matrix_t val = 0);
    Matrix &init(int rows, int cols, matrix_t val = 0);

    // inplace methods
    Matrix operator+=(const Matrix &other);
    Matrix operator*=(const matrix_t &value);
    Matrix operator*=(const Matrix &other);

    // Standard operators
    Matrix operator+(const Matrix &other);
    Matrix operator+(const matrix_t &value);
    Matrix operator-(const Matrix &other);
    Matrix operator-(const matrix_t &value);
    Matrix operator*(const Matrix &other);
    Matrix operator*(const matrix_t &value);
    Matrix dot_multiply(const Matrix &other);
    Matrix transpose();

    // Getter and setter
    // element-wise
    matrix_t get(int row, int col) const;
    void set(int row, int col, matrix_t value);
    // matrix wide getter
    std::vector<matrix_t> data();
    const std::vector<matrix_t> data() const;
    int rows() const;
    int cols() const;

    // convenience methods
    Matrix random(matrix_t low = 0, matrix_t high = 1);
    Matrix fill(matrix_t value);
    matrix_t sum() const;
    void print() const;

    // private data members
private:
    int _rows;
    int _cols;
    std::vector<matrix_t> _data;
};

class Eye : public Matrix
{
public:
    Eye(int rows, int cols);
};
#endif