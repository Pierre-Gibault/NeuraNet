#ifndef NEURANET_MNIST_LOADER_HPP
#define NEURANET_MNIST_LOADER_HPP

#include "matrix.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

// Jeu de données MNIST chargé en mémoire.
struct MnistDataset {
    // Matrice d'images normalisées (pixels x échantillons).
    Matrix images;
    // Tableau d'étiquettes (0..9), une étiquette par échantillon.
    std::uint8_t* labels;
    // Nombre d'échantillons effectivement chargés.
    std::size_t count;
    // Hauteur d'une image en pixels.
    std::size_t rows;
    // Largeur d'une image en pixels.
    std::size_t cols;
};

// Lit un entier 32 bits en big-endian depuis le flux binaire.
inline std::uint32_t readBigEndianUInt32(std::ifstream& stream) {
    unsigned char byte0, byte1, byte2, byte3;
    stream.read((char*)&byte0, 1);
    stream.read((char*)&byte1, 1);
    stream.read((char*)&byte2, 1);
    stream.read((char*)&byte3, 1);

    if (!stream) {
        throw std::runtime_error("Failed to read MNIST header");
    }

    std::uint32_t result;
    result = (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3;
    return result;
}

// Charge les fichiers image/label MNIST et normalise les pixels dans [0, 1].
inline MnistDataset loadMnistDataset(const std::string& imagePath,
                                     const std::string& labelPath,
                                     std::size_t sampleLimit = 0) {
    std::ifstream imageStream(imagePath, std::ios::binary);
    std::ifstream labelStream(labelPath, std::ios::binary);

    if (!imageStream) {
        throw std::runtime_error("Unable to open MNIST image file: " + imagePath);
    }
    if (!labelStream) {
        throw std::runtime_error("Unable to open MNIST label file: " + labelPath);
    }

    const std::uint32_t imageMagic = readBigEndianUInt32(imageStream);
    const std::uint32_t imageCount = readBigEndianUInt32(imageStream);
    const std::uint32_t imageRows = readBigEndianUInt32(imageStream);
    const std::uint32_t imageCols = readBigEndianUInt32(imageStream);

    const std::uint32_t labelMagic = readBigEndianUInt32(labelStream);
    const std::uint32_t labelCount = readBigEndianUInt32(labelStream);

    // Vérifie la validité des en-têtes MNIST.
    if (imageMagic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }
    if (labelMagic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }
    if (imageCount != labelCount) {
        throw std::runtime_error("MNIST image and label counts do not match");
    }

    MnistDataset dataset;
    dataset.images = Matrix(imageRows * imageCols, imageCount);
    dataset.labels = new std::uint8_t[imageCount];
    dataset.count = imageCount;
    dataset.rows = imageRows;
    dataset.cols = imageCols;

    // Limite optionnellement le nombre d'échantillons lus.
    if (sampleLimit > 0 && sampleLimit < dataset.count) {
        dataset.count = sampleLimit;
    }

    std::size_t imageSize = dataset.rows * dataset.cols;
    std::size_t sample;
    std::size_t pixel;
    unsigned char value;
    unsigned char labelValue;
    double pixelDouble;

    // Lit chaque image + son label associé.
    for (sample = 0; sample < dataset.count; ++sample) {
        for (pixel = 0; pixel < imageSize; ++pixel) {
            value = 0;
            imageStream.read((char*)&value, 1);
            if (!imageStream) {
                throw std::runtime_error("Unexpected end of MNIST image file");
            }

            // Normalisation du pixel brut [0, 255] -> [0, 1].
            pixelDouble = value / 255.0;
            dataset.images(pixel, sample) = pixelDouble;
        }

        labelValue = 0;
        labelStream.read((char*)&labelValue, 1);
        if (!labelStream) {
            throw std::runtime_error("Unexpected end of MNIST label file");
        }
        dataset.labels[sample] = labelValue;
    }

    return dataset;
}

#endif