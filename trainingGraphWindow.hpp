#ifndef NEURANET_TRAININGGRAPHWINDOW_HPP
#define NEURANET_TRAININGGRAPHWINDOW_HPP

#include <SFML/Graphics.hpp>

#include <cstddef>
#include <memory>

class TrainingGraphWindow {
public:
    explicit TrainingGraphWindow(std::size_t totalPoints);

    bool isOpen() const;
    void processEvents();
    void addSample(std::size_t sampleIndex, double loss);

private:
    void drawLegend(float x, float y, const sf::Color& color);
    void renderPanel(const sf::FloatRect& panelRect,
                     const sf::Color& lineColor,
                     const double* values,
                     double maxValue);
    void render();
    double computeMaxLoss() const;

    sf::RenderWindow window_;
    std::size_t totalPoints_;
    std::size_t visiblePoints_ = 0;
    bool closedByUser_ = false;
    std::unique_ptr<double[]> lossValues_;
};

#endif
