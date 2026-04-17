#ifndef NEURANET_MATRIX_HPP
#define NEURANET_MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>

// Matrice dense minimaliste pour les opérations du réseau de neurones.
class Matrix {
public:
    // Construit une matrice vide (0x0).
    Matrix() {
        rows_ = 0;
        cols_ = 0;
        data_ = nullptr;
    }

    // Construit une matrice (rows x cols) initialisée avec une valeur constante.
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

    // Constructeur de copie (copie profonde du tampon).
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

    // Libère la mémoire du tampon interne.
    ~Matrix() {
        if (data_ != nullptr) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    // Opérateur d'affectation avec copie profonde.
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

    // Accesseurs de dimensions.
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return rows_ * cols_; }

    // Accès en écriture à un élément (avec vérification par assert).
    double& operator()(std::size_t row, std::size_t col) {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    // Accès en lecture à un élément (avec vérification par assert).
    const double& operator()(std::size_t row, std::size_t col) const {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    // Accès brut au tampon interne (interopération / optimisation).
    double* raw() { return data_; }
    const double* raw() const { return data_; }

    // Remplit tous les éléments avec la même valeur.
    void fill(double value) {
        for (std::size_t index = 0; index < size(); ++index) {
            data_[index] = value;
        }
    }

    // Extrait une colonne de la matrice sous forme de vecteur colonne.
    Matrix column(std::size_t columnIndex) const {
        Matrix result(rows_, 1);
        std::size_t row;
        for (row = 0; row < rows_; ++row) {
            result(row, 0) = (*this)(row, columnIndex);
        }
        return result;
    }

    // Retourne la transposée de la matrice.
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

    // Opérateurs arithmétiques (formes non mutables).
    Matrix operator+(const Matrix& other) const { return add(*this, other); }
    Matrix operator-(const Matrix& other) const { return subtract(*this, other); }
    Matrix operator*(double scalar) const { return scalarMultiply(*this, scalar); }

    // Opérateurs arithmétiques en place.
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

    // Applique la fonction sigmoïde élément par élément.
    void applySigmoid() {
        for (std::size_t index = 0; index < size(); ++index) {
            double value = data_[index];
            data_[index] = 1.0 / (1.0 + std::exp(-value));
        }
    }

    // Renvoie la dérivée de la sigmoïde en supposant des activations déjà sigmoïdées.
    Matrix getSigmoidDerivative() const {
        Matrix result(rows_, cols_);
        for (std::size_t index = 0; index < size(); ++index) {
            double a = data_[index];
            result.data_[index] = a * (1.0 - a);
        }
        return result;
    }

    // Renvoie l'indice du plus grand élément (argmax).
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

    // Fabrique une matrice nulle.
    static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols, 0.0);
    }

    // Fabrique une matrice aléatoire uniforme dans [minValue, maxValue].
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

    // Construit un vecteur colonne à partir d'un tableau C.
    static Matrix fromArray(const double* values, std::size_t count) {
        Matrix result(count, 1);
        for (std::size_t index = 0; index < count; ++index) {
            result(index, 0) = values[index];
        }
        return result;
    }

    // Addition matricielle élément par élément.
    static Matrix add(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, left.cols_);
        std::size_t index;
        for (index = 0; index < left.size(); ++index) {
            result.data_[index] = left.data_[index] + right.data_[index];
        }
        return result;
    }

    // Soustraction matricielle élément par élément.
    static Matrix subtract(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, left.cols_);
        std::size_t index;
        for (index = 0; index < left.size(); ++index) {
            result.data_[index] = left.data_[index] - right.data_[index];
        }
        return result;
    }

    // Produit de Hadamard (multiplication élément par élément).
    static Matrix hadamard(const Matrix& left, const Matrix& right) {
        Matrix result(left.rows_, left.cols_);
        std::size_t index;
        for (index = 0; index < left.size(); ++index) {
            result.data_[index] = left.data_[index] * right.data_[index];
        }
        return result;
    }

    // Multiplication d'une matrice par un scalaire.
    static Matrix scalarMultiply(const Matrix& matrix, double scalar) {
        Matrix result(matrix.rows_, matrix.cols_);
        for (std::size_t index = 0; index < matrix.size(); ++index) {
            result.data_[index] = matrix.data_[index] * scalar;
        }
        return result;
    }

    // Produit matriciel classique (left.rows x right.cols).
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

    // Softmax colonne par colonne (utile pour les probabilités de sortie).
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
    // Nombre de lignes.
    std::size_t rows_;
    // Nombre de colonnes.
    std::size_t cols_;
    // Tampon contigu stockant les valeurs ligne par ligne.
    double* data_;
};

#endif