#include "sevenSegmentDigit.hpp"

#include <algorithm>

SevenSegmentDigit::SevenSegmentDigit(sf::Vector2f position, sf::Vector2f size)
    : topLeft_(position), size_(size) {}

void SevenSegmentDigit::draw(sf::RenderTarget& target, int digit, sf::Color onColor, sf::Color offColor) const {
    bool segments[10][7];
    segments[0][0] = true;  segments[0][1] = true;  segments[0][2] = true;  segments[0][3] = true;  segments[0][4] = true;  segments[0][5] = true;  segments[0][6] = false;
    segments[1][0] = false; segments[1][1] = true;  segments[1][2] = true;  segments[1][3] = false; segments[1][4] = false; segments[1][5] = false; segments[1][6] = false;
    segments[2][0] = true;  segments[2][1] = true;  segments[2][2] = false; segments[2][3] = true;  segments[2][4] = true;  segments[2][5] = false; segments[2][6] = true;
    segments[3][0] = true;  segments[3][1] = true;  segments[3][2] = true;  segments[3][3] = true;  segments[3][4] = false; segments[3][5] = false; segments[3][6] = true;
    segments[4][0] = false; segments[4][1] = true;  segments[4][2] = true;  segments[4][3] = false; segments[4][4] = false; segments[4][5] = true;  segments[4][6] = true;
    segments[5][0] = true;  segments[5][1] = false; segments[5][2] = true;  segments[5][3] = true;  segments[5][4] = false; segments[5][5] = true;  segments[5][6] = true;
    segments[6][0] = true;  segments[6][1] = false; segments[6][2] = true;  segments[6][3] = true;  segments[6][4] = true;  segments[6][5] = true;  segments[6][6] = true;
    segments[7][0] = true;  segments[7][1] = true;  segments[7][2] = true;  segments[7][3] = false; segments[7][4] = false; segments[7][5] = false; segments[7][6] = false;
    segments[8][0] = true;  segments[8][1] = true;  segments[8][2] = true;  segments[8][3] = true;  segments[8][4] = true;  segments[8][5] = true;  segments[8][6] = true;
    segments[9][0] = true;  segments[9][1] = true;  segments[9][2] = true;  segments[9][3] = true;  segments[9][4] = false; segments[9][5] = true;  segments[9][6] = true;

    if (digit < 0) digit = 0;
    if (digit > 9) digit = 9;

    float w = size_.x;
    float h = size_.y;
    float thickness = 4.f;
    if (std::min(w, h) * 0.12f > thickness) {
        thickness = std::min(w, h) * 0.12f;
    }
    float segmentWidth = w - 2.f * thickness;
    float segmentHeight = (h - 3.f * thickness) * 0.5f;

    sf::RectangleShape rect;
    rect.setFillColor(segments[digit][0] ? onColor : offColor);
    rect.setSize(sf::Vector2f(segmentWidth, thickness));
    rect.setPosition(topLeft_.x + thickness, topLeft_.y + 0.f);
    target.draw(rect);

    rect.setFillColor(segments[digit][1] ? onColor : offColor);
    rect.setSize(sf::Vector2f(thickness, segmentHeight));
    rect.setPosition(topLeft_.x + w - thickness, topLeft_.y + thickness);
    target.draw(rect);

    rect.setFillColor(segments[digit][2] ? onColor : offColor);
    rect.setSize(sf::Vector2f(thickness, segmentHeight));
    rect.setPosition(topLeft_.x + w - thickness, topLeft_.y + thickness + segmentHeight + thickness);
    target.draw(rect);

    rect.setFillColor(segments[digit][3] ? onColor : offColor);
    rect.setSize(sf::Vector2f(segmentWidth, thickness));
    rect.setPosition(topLeft_.x + thickness, topLeft_.y + h - thickness);
    target.draw(rect);

    rect.setFillColor(segments[digit][4] ? onColor : offColor);
    rect.setSize(sf::Vector2f(thickness, segmentHeight));
    rect.setPosition(topLeft_.x + 0.f, topLeft_.y + thickness + segmentHeight + thickness);
    target.draw(rect);

    rect.setFillColor(segments[digit][5] ? onColor : offColor);
    rect.setSize(sf::Vector2f(thickness, segmentHeight));
    rect.setPosition(topLeft_.x + 0.f, topLeft_.y + thickness);
    target.draw(rect);

    rect.setFillColor(segments[digit][6] ? onColor : offColor);
    rect.setSize(sf::Vector2f(segmentWidth, thickness));
    rect.setPosition(topLeft_.x + thickness, topLeft_.y + thickness + segmentHeight);
    target.draw(rect);
}
