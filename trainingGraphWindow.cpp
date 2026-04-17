#include "trainingGraphWindow.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

// Initialise la fenêtre de suivi de perte et alloue le buffer des points.
TrainingGraphWindow::TrainingGraphWindow(std::size_t totalPoints)
        : window_(
#if defined(SFML_VERSION_MAJOR) && SFML_VERSION_MAJOR >= 3
                    sf::VideoMode({1100u, 720u}),
#else
                    sf::VideoMode(1100, 720),
#endif
                    "NeuraNet - Training"),
      totalPoints_(totalPoints),
      lossValues_(totalPoints > 0 ? new double[totalPoints] : nullptr) {
    window_.setFramerateLimit(30);
    for (std::size_t index = 0; index < totalPoints_; ++index) {
        lossValues_[index] = 0.0;
    }
}

// Indique si la fenêtre est utilisable pour de nouveaux rendus.
bool TrainingGraphWindow::isOpen() const {
    return window_.isOpen() && !closedByUser_;
}

// Vide la file d'événements SFML (notamment fermeture utilisateur).
void TrainingGraphWindow::processEvents() {
#if defined(SFML_VERSION_MAJOR) && SFML_VERSION_MAJOR >= 3
    while (const auto event = window_.pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            window_.close();
            closedByUser_ = true;
        }
    }
#else
    sf::Event event;
    while (window_.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window_.close();
            closedByUser_ = true;
        }
    }
#endif
}

// Ajoute une valeur de perte et rafraîchit immédiatement l'affichage.
void TrainingGraphWindow::addSample(std::size_t sampleIndex, double loss) {
    if (!isOpen() || sampleIndex >= totalPoints_) {
        return;
    }

    lossValues_[sampleIndex] = loss;
    if (sampleIndex + 1 > visiblePoints_) {
        visiblePoints_ = sampleIndex + 1;
    }

    // Met à jour le titre de fenêtre avec progression et perte courante.
    std::ostringstream title;
    title << "NeuraNet - Training sample " << (sampleIndex + 1) << '/' << totalPoints_ << " | loss: " << std::fixed << std::setprecision(6) << loss;
    window_.setTitle(title.str());

    render();
    processEvents();
}

// Dessine un carré de légende coloré.
void TrainingGraphWindow::drawLegend(float x, float y, const sf::Color& color) {
    sf::RectangleShape swatch({18.f, 18.f});
    swatch.setPosition({x, y});
    swatch.setFillColor(color);
    window_.draw(swatch);
}

// Dessine un panneau de graphe (cadre, axes, courbe et points).
void TrainingGraphWindow::renderPanel(const sf::FloatRect& panelRect,
                                      const sf::Color& lineColor,
                                      const double* values,
                                      double maxValue) {
#if defined(SFML_VERSION_MAJOR) && SFML_VERSION_MAJOR >= 3
    const sf::Vector2f panelPosition = panelRect.position;
    const sf::Vector2f panelSize = panelRect.size;
#else
    const sf::Vector2f panelPosition(panelRect.left, panelRect.top);
    const sf::Vector2f panelSize(panelRect.width, panelRect.height);
#endif

    sf::RectangleShape background(panelSize);
    background.setPosition(panelPosition);
    background.setFillColor(sf::Color(28, 28, 32));
    background.setOutlineThickness(2.f);
    background.setOutlineColor(sf::Color(110, 110, 120));
    window_.draw(background);

    sf::RectangleShape header({panelSize.x, 30.f});
    header.setPosition(panelPosition);
    header.setFillColor(sf::Color(42, 42, 48));
    window_.draw(header);

    drawLegend(panelPosition.x + 16.f, panelPosition.y + 6.f, lineColor);

    const float plotLeft = panelPosition.x + 40.f;
    const float plotTop = panelPosition.y + 42.f;
    const float plotWidth = panelSize.x - 60.f;
    const float plotHeight = panelSize.y - 70.f;
    const float plotBottom = plotTop + plotHeight;

    sf::RectangleShape xAxis({plotWidth, 1.f});
    xAxis.setPosition({plotLeft, plotBottom});
    xAxis.setFillColor(sf::Color(140, 140, 140));
    window_.draw(xAxis);

    sf::RectangleShape yAxis({1.f, plotHeight});
    yAxis.setPosition({plotLeft, plotTop});
    yAxis.setFillColor(sf::Color(140, 140, 140));
    window_.draw(yAxis);

    // Évite de tracer une courbe invalide sans données exploitables.
    if (visiblePoints_ < 2 || maxValue <= 0.0) {
        return;
    }

    sf::VertexArray line(
#if defined(SFML_VERSION_MAJOR) && SFML_VERSION_MAJOR >= 3
        sf::PrimitiveType::LineStrip,
#else
        sf::LineStrip,
#endif
        visiblePoints_);
    // Trace la ligne reliant tous les échantillons visibles.
    for (std::size_t index = 0; index < visiblePoints_; ++index) {
        float x_frac = 0.0f;
        if (totalPoints_ > 1) {
            x_frac = static_cast<float>(index) / static_cast<float>(totalPoints_ - 1);
        }
        float x = plotLeft + x_frac * plotWidth;

        float normalized = values[index] / maxValue;
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;
        float y = plotBottom - normalized * plotHeight;

        line[index].position = sf::Vector2f(x, y);
        line[index].color = lineColor;
    }
    window_.draw(line);

    // Ajoute un marqueur circulaire sur chaque point pour améliorer la lisibilité.
    for (std::size_t index = 0; index < visiblePoints_; ++index) {
        const float x = plotLeft + static_cast<float>(index) * plotWidth /
                                    static_cast<float>(std::max<std::size_t>(1, totalPoints_ - 1));
        const float normalized = static_cast<float>(std::clamp(values[index] / maxValue, 0.0, 1.0));
        const float y = plotBottom - normalized * plotHeight;
        sf::CircleShape point(4.f);
        point.setOrigin({4.f, 4.f});
        point.setPosition({x, y});
        point.setFillColor(lineColor);
        window_.draw(point);
    }
}

// Exécute le rendu complet de la fenêtre de suivi.
void TrainingGraphWindow::render() {
    window_.clear(sf::Color(18, 18, 22));

    const sf::FloatRect lossPanel({24.f, 24.f}, {1050.f, 626.f});

    renderPanel(lossPanel, sf::Color(255, 120, 100), lossValues_.get(), computeMaxLoss());

    window_.display();
}

// Calcule la perte maximale visible pour normaliser l'axe vertical.
double TrainingGraphWindow::computeMaxLoss() const {
    double maxLoss = 0.0;
    for (std::size_t index = 0; index < visiblePoints_; ++index) {
        maxLoss = std::max(maxLoss, lossValues_[index]);
    }
    return maxLoss <= 0.0 ? 1.0 : maxLoss;
}
