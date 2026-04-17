#ifndef NEURANET_MNIST_LOADER_HPP
#define NEURANET_MNIST_LOADER_HPP

#include "matrix.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

struct MnistDataset {
    Matrix images;
    std::uint8_t* labels;
    std::size_t count;
    std::size_t rows;
    std::size_t cols;
};

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

    if (sampleLimit > 0 && sampleLimit < dataset.count) {
        dataset.count = sampleLimit;
    }

    std::size_t imageSize = dataset.rows * dataset.cols;
    std::size_t sample;
    std::size_t pixel;
    unsigned char value;
    unsigned char labelValue;
    double pixelDouble;

    for (sample = 0; sample < dataset.count; ++sample) {
        for (pixel = 0; pixel < imageSize; ++pixel) {
            value = 0;
            imageStream.read((char*)&value, 1);
            if (!imageStream) {
                throw std::runtime_error("Unexpected end of MNIST image file");
            }

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