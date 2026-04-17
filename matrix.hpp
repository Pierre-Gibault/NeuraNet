#ifndef NEURANET_MATRIX_HPP
#define NEURANET_MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>

class Matrix {
public:
    Matrix() {
        rows_ = 0;
        cols_ = 0;
        data_ = nullptr;
    }

    Matrix(std::size_t rows, std::size_t cols, double value = 0.0) {
        rows_ = rows;
        cols_ = cols;
        if (rows * cols > 0) {
            data_ = new double[rows * cols];
        } else {
            data_ = nullptr;
        }
        fill(value);
    }

    Matrix(const Matrix& other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        if (other.size() > 0) {
            data_ = new double[other.size()];
            for (std::size_t index = 0; index < size(); ++index) {
                data_[index] = other.data_[index];
            }
        } else {
            data_ = nullptr;
        }
    }

    ~Matrix() {
        if (data_ != nullptr) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this == &other) {
            return *this;
        }

        if (data_ != nullptr) {
            delete[] data_;
        }

        rows_ = other.rows_;
        cols_ = other.cols_;
        if (other.size() > 0) {
            data_ = new double[other.size()];
            for (std::size_t index = 0; index < size(); ++index) {
                data_[index] = other.data_[index];
            }
        } else {
            data_ = nullptr;
        }

        return *this;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return rows_ * cols_; }

    double& operator()(std::size_t row, std::size_t col) {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    const double& operator()(std::size_t row, std::size_t col) const {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    double* raw() { return data_; }
    const double* raw() const { return data_; }

    void fill(double value) {
        for (std::size_t index = 0; index < size(); ++index) {
            data_[index] = value;
        }
    }

    Matrix column(std::size_t columnIndex) const {
        Matrix result(rows_, 1);
        std::size_t row;
        for (row = 0; row < rows_; ++row) {
            result(row, 0) = (*this)(row, columnIndex);
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols_, rows_);
        std::size_t row;
        std::size_t col;
        for (row = 0; row < rows_; ++row) {
            for (col = 0; col < cols_; ++col) {
                result(col, row) = (*this)(row, col);
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& other) const { return add(*this, other); }
    Matrix operator-(const Matrix& other) const { return subtract(*this, other); }
    Matrix operator*(double scalar) const { return scalarMultiply(*this, scalar); }

    Matrix& operator+=(const Matrix& other) {
        std::size_t index;
        for (index = 0; index < size(); ++index) {
            data_[index] += other.data_[index];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        std::size_t index;
        for (index = 0; index < size(); ++index) {
            data_[index] -= other.data_[index];
        }
        return *this;
    }

    Matrix& operator*=(double scalar) {
        for (std::size_t index = 0; index < size(); ++index) {
            data_[index] *= scalar;
        }
        return *this;
    }

    void applySigmoid() {
        for (std::size_t index = 0; index < size(); ++index) {
            double value = data_[index];
            data_[index] = 1.0 / (1.0 + std::exp(-value));
        }
    }

    Matrix getSigmoidDerivative() const {
        Matrix result(rows_, cols_);
        for (std::size_t index = 0; index < size(); ++index) {
            double a = data_[index];
            result.data_[index] = a * (1.0 - a);
        }
        return result;
    }

    std::size_t argMax() const {
        std::size_t bestIndex = 0;
        double bestValue = data_[0];
        std::size_t index;
        for (index = 1; index < size(); ++index) {
            if (data_[index] > bestValue) {
                bestValue = data_[index];
                bestIndex = index;
            }
        }
        return bestIndex;
    }

    static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols, 0.0);
    }

    static Matrix random(std::size_t rows, std::size_t cols, double minValue = -0.5, double maxValue = 0.5) {
        Matrix result(rows, cols);
        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        std::uniform_real_distribution<double> distribution(minValue, maxValue);

        std::size_t index;
        for (index = 0; index < result.size(); ++index) {
            result.data_[index] = distribution(generator);
        }
        return result;
    }

    static Matrix fromArray(const double* values, std::size_t count) {
        Matrix result(count, 1);
        for (std::size_t index = 0; index < count; ++index) {
            result(index, 0) = values[index];
        }
        return result;
    }

    static Matrix add(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, left.cols_);
        std::size_t index;
        for (index = 0; index < left.size(); ++index) {
            result.data_[index] = left.data_[index] + right.data_[index];
        }
        return result;
    }

    static Matrix subtract(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, left.cols_);
        std::size_t index;
        for (index = 0; index < left.size(); ++index) {
            result.data_[index] = left.data_[index] - right.data_[index];
        }
        return result;
    }

    static Matrix hadamard(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, left.cols_);
        std::size_t index;
        for (index = 0; index < left.size(); ++index) {
            result.data_[index] = left.data_[index] * right.data_[index];
        }
        return result;
    }

    static Matrix scalarMultiply(const Matrix& matrix, double scalar) {
        Matrix result(matrix.rows_, matrix.cols_);
        for (std::size_t index = 0; index < matrix.size(); ++index) {
            result.data_[index] = matrix.data_[index] * scalar;
        }
        return result;
    }

    static Matrix multiply(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, right.cols_);
        std::size_t row;
        std::size_t col;
        std::size_t index;
        double sum;

        for (row = 0; row < left.rows_; ++row) {
            for (col = 0; col < right.cols_; ++col) {
                sum = 0.0;
                for (index = 0; index < left.cols_; ++index) {
                    sum += left(row, index) * right(index, col);
                }
                result(row, col) = sum;
            }
        }
        return result;
    }

    static Matrix softmax(const Matrix& input) {
        Matrix result(input.rows_, input.cols_);

        for (std::size_t col = 0; col < input.cols_; ++col) {
            double maxValue = input(0, col);
            for (std::size_t row = 1; row < input.rows_; ++row) {
                maxValue = std::max(maxValue, input(row, col));
            }

            double sum = 0.0;
            for (std::size_t row = 0; row < input.rows_; ++row) {
                double exponent = std::exp(input(row, col) - maxValue);
                result(row, col) = exponent;
                sum += exponent;
            }

            if (sum == 0.0) {
                continue;
            }

            for (std::size_t row = 0; row < input.rows_; ++row) {
                result(row, col) /= sum;
            }
        }

        return result;
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    double* data_;
};

#endif