#ifndef NEURANET_NEURALNETWORK_HPP
#define NEURANET_NEURALNETWORK_HPP

#include "mnist_loader.hpp"

#include <cstddef>
#include <string>

class NeuralNetwork {
public:
    NeuralNetwork(std::size_t inputSize = 784,
                  std::size_t hiddenSize = 128,
                  std::size_t outputSize = 10);

    Matrix predict(const Matrix& input) const;
    std::size_t predictClass(const Matrix& input) const;

    void train(const MnistDataset& dataset,
               std::size_t epochs,
               double learningRate,
               bool shuffleSamples = true);

    double evaluate(const MnistDataset& dataset) const;

    bool saveWeights(const std::string& path) const;
    bool loadWeights(const std::string& path);

private:
    void forward(const Matrix& input, Matrix& hiddenActivation, Matrix& outputActivation) const;
    double trainSample(const Matrix& input, std::uint8_t label, double learningRate);
    Matrix oneHot(std::uint8_t label) const;

    std::size_t inputSize_;
    std::size_t hiddenSize_;
    std::size_t outputSize_;

    Matrix weightsInputHidden_;
    Matrix biasHidden_;
    Matrix weightsHiddenOutput_;
    Matrix biasOutput_;
};

int runApplication(int argc, char** argv);

#endif