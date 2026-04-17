#ifndef NEURANET_NEURALNETWORK_HPP
#define NEURANET_NEURALNETWORK_HPP

#include "mnist_loader.hpp"

#include <cstddef>
#include <string>

// Réseau de neurones simple à une couche cachée pour la classification MNIST.
class NeuralNetwork {
public:
    // Initialise les tailles de couches et les poids/biais.
    NeuralNetwork(std::size_t inputSize = 784,
                  std::size_t hiddenSize = 128,
                  std::size_t outputSize = 10);

    // Renvoie les probabilités de sortie pour un vecteur d'entrée.
    Matrix predict(const Matrix& input) const;
    // Renvoie la classe la plus probable (argmax de predict).
    std::size_t predictClass(const Matrix& input) const;

    // Entraîne le réseau sur un dataset complet pendant plusieurs époques.
    void train(const MnistDataset& dataset,
               std::size_t epochs,
               double learningRate,
               bool shuffleSamples = true);

    // Mesure la précision du réseau sur un dataset.
    double evaluate(const MnistDataset& dataset) const;

    // Sérialise les poids et biais vers un fichier binaire.
    bool saveWeights(const std::string& path) const;
    // Recharge les poids et biais depuis un fichier binaire.
    bool loadWeights(const std::string& path);

private:
    // Propagation avant : calcule activations cachées et de sortie.
    void forward(const Matrix& input, Matrix& hiddenActivation, Matrix& outputActivation) const;
    // Entraîne le réseau sur un seul échantillon (backprop + mise à jour).
    double trainSample(const Matrix& input, std::uint8_t label, double learningRate);
    // Convertit un label en vecteur one-hot.
    Matrix oneHot(std::uint8_t label) const;

    // Dimensions des couches.
    std::size_t inputSize_;
    std::size_t hiddenSize_;
    std::size_t outputSize_;

    // Paramètres du modèle.
    Matrix weightsInputHidden_;
    Matrix biasHidden_;
    Matrix weightsHiddenOutput_;
    Matrix biasOutput_;
};

// Point d'entrée de l'application (mode entraînement / dessin).
int runApplication(int argc, char** argv);

#endif