#ifndef NEURANET_TRAININGGRAPHWINDOW_HPP
#define NEURANET_TRAININGGRAPHWINDOW_HPP

#include <SFML/Graphics.hpp>

#include <cstddef>
#include <memory>

// Fenêtre SFML affichant l'évolution de la perte pendant l'entraînement.
class TrainingGraphWindow {
public:
    // Prépare la fenêtre pour un nombre total de points connu.
    explicit TrainingGraphWindow(std::size_t totalPoints);

    // Indique si la fenêtre est encore ouverte.
    bool isOpen() const;
    // Traite les événements système (fermeture, etc.).
    void processEvents();
    // Ajoute un échantillon de perte puis rafraîchit l'affichage.
    void addSample(std::size_t sampleIndex, double loss);

private:
    // Dessine la légende d'un tracé.
    void drawLegend(float x, float y, const sf::Color& color);
    // Dessine un panneau graphique avec une courbe de valeurs.
    void renderPanel(const sf::FloatRect& panelRect,
                     const sf::Color& lineColor,
                     const double* values,
                     double maxValue);
    // Rendu complet de la fenêtre.
    void render();
    // Renvoie la perte maximale visible pour normaliser l'échelle.
    double computeMaxLoss() const;

    // Fenêtre de visualisation.
    sf::RenderWindow window_;
    // Nombre total d'échantillons attendus.
    std::size_t totalPoints_;
    // Nombre d'échantillons déjà reçus.
    std::size_t visiblePoints_ = 0;
    // Mémorise une fermeture volontaire de l'utilisateur.
    bool closedByUser_ = false;
    // Historique des pertes par échantillon.
    std::unique_ptr<double[]> lossValues_;
};

#endif
