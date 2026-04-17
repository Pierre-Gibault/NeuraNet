#include "drawingApp.hpp"

#include "sevenSegmentDigit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

// Initialise la fenêtre, le canevas et les structures de prédiction.
DrawingApp::DrawingApp(NeuralNetwork& network)
    : network_(network),
            window_(
#if defined(SFML_VERSION_MAJOR) && SFML_VERSION_MAJOR >= 3
                    sf::VideoMode({900u, 520u}),
#else
                    sf::VideoMode(900, 520),
#endif
                    "NeuraNet - MNIST drawing"),
      canvas_(gridSize_ * gridSize_, 1),
      predictedProbabilities_(10, 1) {
    window_.setFramerateLimit(60);
    clearCanvas();
}

// Boucle principale : gestion des événements, inférence puis rendu.
void DrawingApp::run() {
    while (window_.isOpen()) {
        handleEvents();
        if (canvasDirty_) {
            updatePrediction();
            canvasDirty_ = false;
        }
        render();
    }
}

// Traite les interactions utilisateur (fermeture, dessin, effacement).
void DrawingApp::handleEvents() {
#if defined(SFML_VERSION_MAJOR) && SFML_VERSION_MAJOR >= 3
    while (const auto event = window_.pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            window_.close();
        } else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
            if (keyPressed->code == sf::Keyboard::Key::C) {
                clearCanvas();
            }
        } else if (const auto* mousePressed = event->getIf<sf::Event::MouseButtonPressed>()) {
            if (mousePressed->button == sf::Mouse::Button::Left) {
                paint(mousePressed->position.x, mousePressed->position.y);
            }
        } else if (const auto* mouseMoved = event->getIf<sf::Event::MouseMoved>()) {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
                paint(mouseMoved->position.x, mouseMoved->position.y);
            }
        }
    }
#else
    sf::Event event;
    while (window_.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window_.close();
        } else if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::C) {
                clearCanvas();
            }
        } else if (event.type == sf::Event::MouseButtonPressed || event.type == sf::Event::MouseMoved) {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                const sf::Vector2i mousePosition = sf::Mouse::getPosition(window_);
                paint(mousePosition.x, mousePosition.y);
            }
        }
    }
#endif
}

// Dépose de l'encre sur le canevas avec un pinceau circulaire.
void DrawingApp::paint(int x, int y) {
    // Ignore les coordonnées en dehors de la zone de dessin.
    if (x < canvasLeft_ || y < canvasTop_) {
        return;
    }

    int localX = x - canvasLeft_;
    int localY = y - canvasTop_;

    if (localX >= gridSize_ * cellSize_ || localY >= gridSize_ * cellSize_) {
        return;
    }

    int centerX = localX / cellSize_;
    int centerY = localY / cellSize_;

    int r;
    int c;
    int targetX;
    int targetY;
    int dist_sq;
    std::size_t index;
    double val;

    // Parcourt le voisinage du centre pour appliquer le pinceau.
    for (r = -brushRadius_; r <= brushRadius_; ++r) {
        for (c = -brushRadius_; c <= brushRadius_; ++c) {
            targetX = centerX + c;
            targetY = centerY + r;
            if (targetX < 0 || targetY < 0 || targetX >= gridSize_ || targetY >= gridSize_) {
                continue;
            }

            dist_sq = r * r + c * c;
            if (dist_sq > brushRadius_ * brushRadius_) {
                continue;
            }

            index = targetY * gridSize_ + targetX;
            // Cumule l'intensité et borne à 1.0.
            val = canvas_(index, 0) + 0.35;
            if (val > 1.0) {
                val = 1.0;
            }
            canvas_(index, 0) = val;
        }
    }

    canvasDirty_ = true;
}

// Réinitialise entièrement le canevas et l'état de prédiction.
void DrawingApp::clearCanvas() {
    canvas_.fill(0.0);
    predictedProbabilities_.fill(0.0);
    predictedDigit_ = 0;
    hasPrediction_ = false;
    canvasDirty_ = true;
}

// Construit un vecteur colonne 784x1 à partir de la grille 28x28.
Matrix DrawingApp::buildInputVector() const {
    Matrix input(784, 1);
    for (int row = 0; row < gridSize_; ++row) {
        for (int col = 0; col < gridSize_; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * gridSize_ + static_cast<std::size_t>(col);
            input(index, 0) = canvas_(index, 0);
        }
    }

    return input;
}

// Lance l'inférence réseau sur le canevas courant.
void DrawingApp::updatePrediction() {
    const Matrix input = buildInputVector();
    predictedProbabilities_ = network_.predict(input);
    predictedDigit_ = static_cast<int>(predictedProbabilities_.argMax());
    hasPrediction_ = true;
}

// Dessine une frame complète de l'interface.
void DrawingApp::render() {
    window_.clear(sf::Color(30, 30, 35));

    drawPredictionPanel();
    drawCanvas();
    window_.display();
}

// Affiche le panneau latéral des classes et probabilités.
void DrawingApp::drawPredictionPanel() {
    sf::RectangleShape panel({280.f, 472.f});
    panel.setPosition({20.f, 24.f});
    panel.setFillColor(sf::Color(44, 44, 50));
    panel.setOutlineThickness(2.f);
    panel.setOutlineColor(sf::Color(120, 120, 130));
    window_.draw(panel);

    const float baseY = 36.f;
    const float rowHeight = 42.f;
    const float rowGap = 4.f;

    // Une ligne par classe (0 à 9).
    for (int index = 0; index < 10; ++index) {
        const float rowY = baseY + index * (rowHeight + rowGap);

        sf::RectangleShape rowBackground({248.f, rowHeight});
        rowBackground.setPosition({36.f, rowY});
        rowBackground.setFillColor(index == predictedDigit_ ? sf::Color(64, 82, 105) : sf::Color(55, 55, 60));
        window_.draw(rowBackground);

        SevenSegmentDigit classDigit({46.f, rowY + 4.f}, {24.f, 34.f});
        classDigit.draw(window_, index, sf::Color(175, 175, 180), sf::Color(72, 72, 74));

        const int percent = static_cast<int>(std::round(predictedProbabilities_(index, 0) * 100.0));
        drawPercentageValue(96.f, rowY + 7.f, percent, index == predictedDigit_);
    }
}

// Dessine une valeur numérique sur 3 digits, suivie d'un symbole de pourcentage stylisé.
void DrawingApp::drawPercentageValue(float x, float y, int percent, bool highlighted) {
    const int boundedPercent = std::clamp(percent, 0, 100);
    const int hundreds = boundedPercent / 100;
    const int tens = (boundedPercent / 10) % 10;
    const int units = boundedPercent % 10;

    const sf::Color activeColor = highlighted ? sf::Color(130, 220, 255) : sf::Color(165, 165, 170);
    const sf::Color inactiveColor = sf::Color(74, 74, 76);

    SevenSegmentDigit digitHundreds({x, y}, {18.f, 28.f});
    SevenSegmentDigit digitTens({x + 24.f, y}, {18.f, 28.f});
    SevenSegmentDigit digitUnits({x + 48.f, y}, {18.f, 28.f});

    digitHundreds.draw(window_, hundreds, activeColor, inactiveColor);
    digitTens.draw(window_, tens, activeColor, inactiveColor);
    digitUnits.draw(window_, units, activeColor, inactiveColor);

    // Dessine un caractère '%' simplifié.
    sf::RectangleShape slash1({4.f, 12.f});
    slash1.setPosition({x + 78.f, y + 2.f});
    slash1.setFillColor(activeColor);
    window_.draw(slash1);

    sf::RectangleShape slash2({4.f, 12.f});
    slash2.setPosition({x + 88.f, y + 14.f});
    slash2.setFillColor(activeColor);
    window_.draw(slash2);
}

// Dessine le cadre et les pixels du canevas en niveaux de gris.
void DrawingApp::drawCanvas() {
    sf::RectangleShape frame({static_cast<float>(gridSize_ * cellSize_ + 8), static_cast<float>(gridSize_ * cellSize_ + 8)});
    frame.setPosition({static_cast<float>(canvasLeft_ - 4), static_cast<float>(canvasTop_ - 4)});
    frame.setFillColor(sf::Color::Transparent);
    frame.setOutlineThickness(2.f);
    frame.setOutlineColor(sf::Color(180, 180, 180));
    window_.draw(frame);

    // Convertit chaque cellule en pixel affiché.
    for (int row = 0; row < gridSize_; ++row) {
        for (int col = 0; col < gridSize_; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * gridSize_ + static_cast<std::size_t>(col);
            const std::uint8_t value = static_cast<std::uint8_t>(std::clamp(canvas_(index, 0), 0.0, 1.0) * 255.0);

            sf::RectangleShape pixel({static_cast<float>(cellSize_ - 1), static_cast<float>(cellSize_ - 1)});
            pixel.setPosition({static_cast<float>(canvasLeft_ + col * cellSize_), static_cast<float>(canvasTop_ + row * cellSize_)});
            pixel.setFillColor(sf::Color(value, value, value));
            window_.draw(pixel);
        }
    }
}
