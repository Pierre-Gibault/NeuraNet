#include "neuralNetwork.hpp"
#include "drawingApp.hpp"
#include "trainingGraphWindow.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace {

// Sigmoïde appliquée élément par élément.
Matrix sigmoid(const Matrix& input) {
    Matrix result(input.rows(), input.cols());
    for (std::size_t i = 0; i < input.size(); ++i) {
        double val = input.raw()[i];
        result.raw()[i] = 1.0 / (1.0 + std::exp(-val));
    }
    return result;
}

// Dérivée de la sigmoïde à partir des activations déjà calculées.
Matrix sigmoidDerivativeFromActivation(const Matrix& activation) {
    Matrix result(activation.rows(), activation.cols());
    for (std::size_t i = 0; i < activation.size(); ++i) {
        double a = activation.raw()[i];
        result.raw()[i] = a * (1.0 - a);
    }
    return result;
}

// Convertit un label scalaire en vecteur one-hot.
Matrix createOneHot(std::uint8_t label, std::size_t outputSize) {
    Matrix result(outputSize, 1);
    if (label < outputSize) {
        result(label, 0) = 1.0;
    }
    return result;
}

} // namespace

// Initialise l'architecture et les paramètres du réseau.
NeuralNetwork::NeuralNetwork(std::size_t inputSize,
                             std::size_t hiddenSize,
                             std::size_t outputSize)
    : inputSize_(inputSize),
      hiddenSize_(hiddenSize),
      outputSize_(outputSize),
      weightsInputHidden_(Matrix::random(hiddenSize, inputSize, -0.1, 0.1)),
      biasHidden_(Matrix::zeros(hiddenSize, 1)),
      weightsHiddenOutput_(Matrix::random(outputSize, hiddenSize, -0.1, 0.1)),
      biasOutput_(Matrix::zeros(outputSize, 1)) {}

// Propagation avant : couches linéaires puis activations.
void NeuralNetwork::forward(const Matrix& input, Matrix& hiddenActivation, Matrix& outputActivation) const {
    Matrix hiddenLinear = Matrix::add(Matrix::multiply(weightsInputHidden_, input), biasHidden_);
    hiddenActivation = sigmoid(hiddenLinear);

    Matrix outputLinear = Matrix::add(Matrix::multiply(weightsHiddenOutput_, hiddenActivation), biasOutput_);
    outputActivation = Matrix::softmax(outputLinear);
}

// Renvoie les probabilités de classes pour une entrée donnée.
Matrix NeuralNetwork::predict(const Matrix& input) const {
    Matrix hiddenActivation;
    Matrix outputActivation;
    forward(input, hiddenActivation, outputActivation);
    return outputActivation;
}

// Renvoie la classe la plus probable.
std::size_t NeuralNetwork::predictClass(const Matrix& input) const {
    return predict(input).argMax();
}

// Effectue un pas d'entraînement complet (forward + backprop + update).
double NeuralNetwork::trainSample(const Matrix& input, std::uint8_t label, double learningRate) {
    Matrix hiddenActivation;
    Matrix outputActivation;
    forward(input, hiddenActivation, outputActivation);

    const Matrix target = oneHot(label);

    // Perte cross-entropy pour un échantillon.
    double sampleLoss = 0.0;
    for (std::size_t row = 0; row < outputSize_; ++row) {
        const double predicted = std::max(1e-12, outputActivation(row, 0));
        sampleLoss -= target(row, 0) * std::log(predicted);
    }

    // Gradient de la couche de sortie (softmax + CE).
    Matrix dZ2 = Matrix::subtract(outputActivation, target);
    Matrix dW2 = Matrix::multiply(dZ2, hiddenActivation.transpose());
    Matrix dB2 = dZ2;

    Matrix dA1 = Matrix::multiply(weightsHiddenOutput_.transpose(), dZ2);
    Matrix dZ1 = Matrix::hadamard(dA1, sigmoidDerivativeFromActivation(hiddenActivation));
    Matrix dW1 = Matrix::multiply(dZ1, input.transpose());
    Matrix dB1 = dZ1;

    // Descente de gradient sur poids et biais.
    weightsHiddenOutput_ -= dW2 * learningRate;
    biasOutput_ -= dB2 * learningRate;
    weightsInputHidden_ -= dW1 * learningRate;
    biasHidden_ -= dB1 * learningRate;

    return sampleLoss;
}

// Encodage one-hot d'un label pour la sortie du réseau.
Matrix NeuralNetwork::oneHot(std::uint8_t label) const {
    return createOneHot(label, outputSize_);
}

// Boucle d'entraînement sur plusieurs époques.
void NeuralNetwork::train(const MnistDataset& dataset,
                          std::size_t epochs,
                          double learningRate,
                          bool shuffleSamples) {
    if (dataset.count == 0) {
        throw std::runtime_error("Cannot train on an empty dataset");
    }

    std::size_t* order = new std::size_t[dataset.count];
    std::size_t i;
    for (i = 0; i < dataset.count; ++i) {
        order[i] = i;
    }

    // Générateur pseudo-aléatoire pour le mélange des échantillons.
    std::mt19937 generator;

    std::size_t graphStride = 100;
    std::size_t totalPoints = 0;
    if (epochs > 0 && dataset.count > 0) {
        totalPoints = (epochs * dataset.count + graphStride - 1) / graphStride;
    }

    TrainingGraphWindow* trainingGraph = nullptr;
    if (totalPoints > 0) {
        trainingGraph = new TrainingGraphWindow(totalPoints);
    }

    std::size_t plottedPointIndex = 0;
    std::size_t epoch;
    std::size_t index;
    std::size_t sampleIndex;
    std::size_t dataIndex;
    double totalLoss;
    double sampleLoss;
    std::size_t swapIndex;
    Matrix input;

    // Itère sur les époques.
    for (epoch = 0; epoch < epochs; ++epoch) {
        // Mélange de l'ordre des échantillons pour améliorer la généralisation.
        if (shuffleSamples) {
            for (index = dataset.count; index > 1; --index) {
                std::uniform_int_distribution<std::size_t> distribution(0, index - 1);
                swapIndex = distribution(generator);
                std::size_t temp = order[index - 1];
                order[index - 1] = order[swapIndex];
                order[swapIndex] = temp;
            }
        }

        totalLoss = 0.0;

        // Itère sur tous les échantillons du dataset.
        for (sampleIndex = 0; sampleIndex < dataset.count; ++sampleIndex) {
            dataIndex = shuffleSamples ? order[sampleIndex] : sampleIndex;
            input = dataset.images.column(dataIndex);
            std::uint8_t label = dataset.labels[dataIndex];

            sampleLoss = trainSample(input, label, learningRate);

            totalLoss += sampleLoss;

            // Met à jour le graphe de perte avec sous-échantillonnage.
            if (trainingGraph != nullptr && trainingGraph->isOpen() && (plottedPointIndex % graphStride == 0)) {
                trainingGraph->addSample(plottedPointIndex / graphStride, sampleLoss);
            }
            ++plottedPointIndex;
        }

        // Affiche le suivi console de l'entraînement.
        double averageLoss = totalLoss / (double)dataset.count;
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
              << " - loss: " << averageLoss
                  << " - accuracy: " << evaluate(dataset) * 100.0 << "%\n";
    }

    // Nettoyage manuel des allocations temporaires.
    if (trainingGraph != nullptr) {
        delete trainingGraph;
    }
    delete[] order;
}

// Évalue la précision du réseau sur un dataset donné.
double NeuralNetwork::evaluate(const MnistDataset& dataset) const {
    if (dataset.count == 0) {
        return 0.0;
    }

    std::size_t correct = 0;
    std::size_t sample;
    Matrix input;
    std::size_t predicted;

    for (sample = 0; sample < dataset.count; ++sample) {
        input = dataset.images.column(sample);
        predicted = predictClass(input);
        if (predicted == dataset.labels[sample]) {
            ++correct;
        }
    }

    return (double)correct / (double)dataset.count;
}

// Sauvegarde les paramètres du modèle au format binaire propriétaire.
bool NeuralNetwork::saveWeights(const std::string& path) const {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }

    char magic[4];
    magic[0] = 'N';
    magic[1] = 'R';
    magic[2] = 'N';
    magic[3] = '1';
    // Écrit un en-tête magique pour vérifier l'intégrité au chargement.
    stream.write(magic, 4);

    std::uint64_t inputSize = inputSize_;
    std::uint64_t hiddenSize = hiddenSize_;
    std::uint64_t outputSize = outputSize_;
    stream.write((char*)&inputSize, sizeof(inputSize));
    stream.write((char*)&hiddenSize, sizeof(hiddenSize));
    stream.write((char*)&outputSize, sizeof(outputSize));

    std::streamsize size1 = weightsInputHidden_.size() * sizeof(double);
    stream.write((char*)weightsInputHidden_.raw(), size1);
    
    std::streamsize size2 = biasHidden_.size() * sizeof(double);
    stream.write((char*)biasHidden_.raw(), size2);
    
    std::streamsize size3 = weightsHiddenOutput_.size() * sizeof(double);
    stream.write((char*)weightsHiddenOutput_.raw(), size3);
    
    std::streamsize size4 = biasOutput_.size() * sizeof(double);
    stream.write((char*)biasOutput_.raw(), size4);

    if (stream) return true;
    return false;
}

// Recharge les paramètres du modèle depuis un fichier binaire.
bool NeuralNetwork::loadWeights(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }

    char magic[4];
    stream.read(magic, 4);
    if (!stream || magic[0] != 'N' || magic[1] != 'R' || magic[2] != 'N' || magic[3] != '1') {
        return false;
    }

    std::uint64_t inputSize = 0;
    std::uint64_t hiddenSize = 0;
    std::uint64_t outputSize = 0;
    stream.read((char*)&inputSize, sizeof(inputSize));
    stream.read((char*)&hiddenSize, sizeof(hiddenSize));
    stream.read((char*)&outputSize, sizeof(outputSize));

    if (!stream || inputSize != inputSize_ || hiddenSize != hiddenSize_ || outputSize != outputSize_) {
        return false;
    }

    std::streamsize size1 = weightsInputHidden_.size() * sizeof(double);
    stream.read((char*)weightsInputHidden_.raw(), size1);
    
    std::streamsize size2 = biasHidden_.size() * sizeof(double);
    stream.read((char*)biasHidden_.raw(), size2);
    
    std::streamsize size3 = weightsHiddenOutput_.size() * sizeof(double);
    stream.read((char*)weightsHiddenOutput_.raw(), size3);
    
    std::streamsize size4 = biasOutput_.size() * sizeof(double);
    stream.read((char*)biasOutput_.raw(), size4);

    if (stream) return true;
    return false;
}

namespace {

// Affiche l'aide de la ligne de commande.
void printUsage() {
    std::cout << "NeuraNet usage:\n"
              << "  neuranet draw [weights.bin]\n"
              << "  neuranet train <mnist_dir> [epochs] [learning_rate] [weights.bin]\n"
              << "  neuranet test <mnist_dir> [weights.bin]\n";
}

// Concatène un chemin de dossier et un nom de fichier.
std::string joinPath(const std::string& directory, const std::string& name) {
    if (directory.empty() || directory.back() == '/') {
        return directory + name;
    }
    return directory + "/" + name;
}

// Charge l'ensemble d'entraînement MNIST.
MnistDataset loadTrainSet(const std::string& directory, std::size_t limit = 0) {
    return loadMnistDataset(joinPath(directory, "train-images.idx3-ubyte"),
                            joinPath(directory, "train-labels.idx1-ubyte"),
                            limit);
}

// Charge l'ensemble de test MNIST.
MnistDataset loadTestSet(const std::string& directory, std::size_t limit = 0) {
    return loadMnistDataset(joinPath(directory, "t10k-images.idx3-ubyte"),
                            joinPath(directory, "t10k-labels.idx1-ubyte"),
                            limit);
}

// Lance l'interface de dessin avec des poids chargés ou aléatoires.
void runDrawMode(const std::string& weightsPath) {
    NeuralNetwork network;
    if (!weightsPath.empty() && network.loadWeights(weightsPath)) {
        std::cout << "Loaded weights from " << weightsPath << '\n';
    } else {
        std::cout << "Starting with random weights. Train first for better predictions.\n";
    }

    DrawingApp app(network);
    app.run();
}

} // namespace

// Routeur principal des commandes CLI.
int runApplication(int argc, char** argv) {
    // Sans argument : démarre directement le mode dessin avec poids par défaut.
    if (argc <= 1) {
        runDrawMode("weights.bin");
        return 0;
    }

    const std::string command(argv[1]);

    if (command == "draw") {
        const std::string weightsPath = argc > 2 ? argv[2] : "weights.bin";
        runDrawMode(weightsPath);
        return 0;
    }

    // Mode entraînement sur dataset MNIST.
    if (command == "train") {
        if (argc < 3) {
            printUsage();
            return 1;
        }

        const std::string datasetDirectory = argv[2];
        const std::size_t epochs = argc > 3 ? static_cast<std::size_t>(std::stoul(argv[3])) : 3;
        const double learningRate = argc > 4 ? std::stod(argv[4]) : 0.01;
        const std::string weightsPath = argc > 5 ? argv[5] : "weights.bin";

        NeuralNetwork network;
        const MnistDataset trainSet = loadTrainSet(datasetDirectory);
        network.train(trainSet, epochs, learningRate);

        network.saveWeights(weightsPath);
        std::cout << "Saved weights to " << weightsPath << '\n';
        return 0;
    }

    // Mode évaluation sur l'ensemble de test MNIST.
    if (command == "test") {
        if (argc < 3) {
            printUsage();
            return 1;
        }

        const std::string datasetDirectory = argv[2];
        const std::string weightsPath = argc > 3 ? argv[3] : "weights.bin";

        NeuralNetwork network;
        network.loadWeights(weightsPath);

        const MnistDataset testSet = loadTestSet(datasetDirectory);
        const double accuracy = network.evaluate(testSet) * 100.0;
        std::cout << "Test accuracy: " << accuracy << "%\n";
        return 0;
    }

    printUsage();
    return 1;
}
