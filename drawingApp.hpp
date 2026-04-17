#ifndef NEURANET_DRAWINGAPP_HPP
#define NEURANET_DRAWINGAPP_HPP

#include "neuralNetwork.hpp"

#include <SFML/Graphics.hpp>

// Application de dessin interactive permettant de tester le réseau sur des chiffres manuscrits.
class DrawingApp {
public:
    // Construit l'application avec une référence vers un réseau déjà initialisé.
    explicit DrawingApp(NeuralNetwork& network);
    // Lance la boucle principale d'affichage et d'interaction.
    void run();

private:
    // Gère les événements clavier/souris de la fenêtre.
    void handleEvents();
    // Applique le pinceau autour de la position demandée.
    void paint(int x, int y);
    // Efface complètement la zone de dessin.
    void clearCanvas();
    // Convertit le canevas courant en vecteur d'entrée pour le réseau.
    Matrix buildInputVector() const;
    // Recalcule la prédiction si le canevas a été modifié.
    void updatePrediction();
    // Dessine une frame complète (UI + canevas + prédiction).
    void render();
    // Affiche le panneau des probabilités de classes.
    void drawPredictionPanel();
    // Affiche un pourcentage numérique dans le panneau de prédiction.
    void drawPercentageValue(float x, float y, int percent, bool highlighted);
    // Dessine la grille 28x28 dans la fenêtre.
    void drawCanvas();

    // Réseau de neurones utilisé pour inférer le chiffre.
    NeuralNetwork& network_;
    // Fenêtre SFML principale de l'application.
    sf::RenderWindow window_;
    // Canevas 28x28 contenant des intensités normalisées [0, 1].
    Matrix canvas_;
    // Indique si le canevas a changé depuis la dernière prédiction.
    bool canvasDirty_ = true;
    // Rayon du pinceau (en cellules du canevas).
    int brushRadius_ = 1;
    // Indique si une prédiction valide est disponible.
    bool hasPrediction_ = false;
    // Classe prédite (chiffre 0..9).
    int predictedDigit_ = 0;
    // Probabilités prédites pour les 10 classes.
    Matrix predictedProbabilities_;

    // Taille de l'image MNIST (28x28).
    static constexpr int gridSize_ = 28;
    // Taille en pixels d'une cellule de la grille affichée.
    static constexpr int cellSize_ = 16;
    // Position X du canevas dans la fenêtre.
    static constexpr int canvasLeft_ = 330;
    // Position Y du canevas dans la fenêtre.
    static constexpr int canvasTop_ = 36;
};

#endif
