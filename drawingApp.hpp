#ifndef NEURANET_DRAWINGAPP_HPP
#define NEURANET_DRAWINGAPP_HPP

#include "neuralNetwork.hpp"

#include <SFML/Graphics.hpp>

class DrawingApp {
public:
    explicit DrawingApp(NeuralNetwork& network);
    void run();

private:
    void handleEvents();
    void paint(int x, int y);
    void clearCanvas();
    Matrix buildInputVector() const;
    void updatePrediction();
    void render();
    void drawPredictionPanel();
    void drawPercentageValue(float x, float y, int percent, bool highlighted);
    void drawCanvas();

    NeuralNetwork& network_;
    sf::RenderWindow window_;
    Matrix canvas_;
    bool canvasDirty_ = true;
    int brushRadius_ = 1;
    bool hasPrediction_ = false;
    int predictedDigit_ = 0;
    Matrix predictedProbabilities_;

    static constexpr int gridSize_ = 28;
    static constexpr int cellSize_ = 16;
    static constexpr int canvasLeft_ = 330;
    static constexpr int canvasTop_ = 36;
};

#endif
