#include "trainingGraphWindow.hpp"

#include <algorithm>
#include <sstream>

TrainingGraphWindow::TrainingGraphWindow(std::size_t totalPoints)
    : window_(sf::VideoMode(1100, 720), "NeuraNet - Training"),
      totalPoints_(totalPoints),
      lossValues_(totalPoints > 0 ? new double[totalPoints] : nullptr) {
    window_.setFramerateLimit(30);
    for (std::size_t index = 0; index < totalPoints_; ++index) {
        lossValues_[index] = 0.0;
    }
}

bool TrainingGraphWindow::isOpen() const {
    return window_.isOpen() && !closedByUser_;
}

void TrainingGraphWindow::processEvents() {
    sf::Event event;
    while (window_.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window_.close();
            closedByUser_ = true;
        }
    }
}

void TrainingGraphWindow::addSample(std::size_t sampleIndex, double loss) {
    if (!isOpen() || sampleIndex >= totalPoints_) {
        return;
    }

    lossValues_[sampleIndex] = loss;
    if (sampleIndex + 1 > visiblePoints_) {
        visiblePoints_ = sampleIndex + 1;
    }

    char buffer[256];
    sprintf(buffer, "NeuraNet - Training sample %lu/%lu | loss: %f", sampleIndex + 1, totalPoints_, loss);
    window_.setTitle(buffer);

    render();
    processEvents();
}

void TrainingGraphWindow::drawLegend(float x, float y, const sf::Color& color) {
    sf::RectangleShape swatch({18.f, 18.f});
    swatch.setPosition(x, y);
    swatch.setFillColor(color);
    window_.draw(swatch);
}

void TrainingGraphWindow::renderPanel(const sf::FloatRect& panelRect,
                                      const sf::Color& lineColor,
                                      const double* values,
                                      double maxValue) {
    sf::RectangleShape background({panelRect.width, panelRect.height});
    background.setPosition(panelRect.left, panelRect.top);
    background.setFillColor(sf::Color(28, 28, 32));
    background.setOutlineThickness(2.f);
    background.setOutlineColor(sf::Color(110, 110, 120));
    window_.draw(background);

    sf::RectangleShape header({panelRect.width, 30.f});
    header.setPosition(panelRect.left, panelRect.top);
    header.setFillColor(sf::Color(42, 42, 48));
    window_.draw(header);

    drawLegend(panelRect.left + 16.f, panelRect.top + 6.f, lineColor);

    const float plotLeft = panelRect.left + 40.f;
    const float plotTop = panelRect.top + 42.f;
    const float plotWidth = panelRect.width - 60.f;
    const float plotHeight = panelRect.height - 70.f;
    const float plotBottom = plotTop + plotHeight;

    sf::RectangleShape xAxis({plotWidth, 1.f});
    xAxis.setPosition(plotLeft, plotBottom);
    xAxis.setFillColor(sf::Color(140, 140, 140));
    window_.draw(xAxis);

    sf::RectangleShape yAxis({1.f, plotHeight});
    yAxis.setPosition(plotLeft, plotTop);
    yAxis.setFillColor(sf::Color(140, 140, 140));
    window_.draw(yAxis);

    if (visiblePoints_ < 2 || maxValue <= 0.0) {
        return;
    }

    sf::VertexArray line(sf::LineStrip, visiblePoints_);
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

    for (std::size_t index = 0; index < visiblePoints_; ++index) {
        const float x = plotLeft + static_cast<float>(index) * plotWidth /
                                    static_cast<float>(std::max<std::size_t>(1, totalPoints_ - 1));
        const float normalized = static_cast<float>(std::clamp(values[index] / maxValue, 0.0, 1.0));
        const float y = plotBottom - normalized * plotHeight;
        sf::CircleShape point(4.f);
        point.setOrigin(4.f, 4.f);
        point.setPosition(x, y);
        point.setFillColor(lineColor);
        window_.draw(point);
    }
}

void TrainingGraphWindow::render() {
    window_.clear(sf::Color(18, 18, 22));

    const sf::FloatRect lossPanel(24.f, 24.f, 1050.f, 626.f);

    renderPanel(lossPanel, sf::Color(255, 120, 100), lossValues_.get(), computeMaxLoss());

    window_.display();
}

double TrainingGraphWindow::computeMaxLoss() const {
    double maxLoss = 0.0;
    for (std::size_t index = 0; index < visiblePoints_; ++index) {
        maxLoss = std::max(maxLoss, lossValues_[index]);
    }
    return maxLoss <= 0.0 ? 1.0 : maxLoss;
}
