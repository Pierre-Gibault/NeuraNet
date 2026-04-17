#ifndef NEURANET_SEVENSEGMENTDIGIT_HPP
#define NEURANET_SEVENSEGMENTDIGIT_HPP

#include <SFML/Graphics.hpp>

class SevenSegmentDigit {
public:
    SevenSegmentDigit(sf::Vector2f position, sf::Vector2f size);
    void draw(sf::RenderTarget& target, int digit, sf::Color onColor, sf::Color offColor) const;

private:
    sf::Vector2f topLeft_;
    sf::Vector2f size_;
};

#endif
