#ifndef NEURANET_SEVENSEGMENTDIGIT_HPP
#define NEURANET_SEVENSEGMENTDIGIT_HPP

#include <SFML/Graphics.hpp>

// Affiche un chiffre avec un rendu "7 segments".
class SevenSegmentDigit {
public:
    // Définit la position et la taille du digit à dessiner.
    SevenSegmentDigit(sf::Vector2f position, sf::Vector2f size);
    // Dessine le chiffre demandé avec des couleurs ON/OFF.
    void draw(sf::RenderTarget& target, int digit, sf::Color onColor, sf::Color offColor) const;

private:
    // Coin supérieur gauche de la zone du digit.
    sf::Vector2f topLeft_;
    // Taille totale allouée au digit.
    sf::Vector2f size_;
};

#endif
