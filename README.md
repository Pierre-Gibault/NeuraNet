# NeuraNet

NeuraNet est un réseau de neurones en C++ pour MNIST, avec une classe `Matrix` maison, la propagation avant, la rétropropagation et une interface SFML pour dessiner un chiffre en temps réel.

Pendant l’entraînement, une fenêtre SFML affiche aussi un graphe live de la perte en ne gardant qu’un point sur 100 propagations avant.

## Idée générale

Le réseau apprend à associer une image de chiffre manuscrit à une sortie parmi 10 classes, de `0` à `9`.

Dans ce projet, une image MNIST est représentée par un vecteur colonne de taille `784`, car l’image fait `28 × 28` pixels :

$$x \in \mathbb{R}^{784 \times 1}$$

Le réseau utilisé ici a 2 couches apprenables :

- une couche cachée de taille `128`
- une couche de sortie de taille `10`

Les poids et biais sont :

$$W_1 \in \mathbb{R}^{128 \times 784}, \quad b_1 \in \mathbb{R}^{128 \times 1}$$

$$W_2 \in \mathbb{R}^{10 \times 128}, \quad b_2 \in \mathbb{R}^{10 \times 1}$$

## Représentation matricielle

La classe `Matrix` remplace `std::vector` pour stocker les données numériques.

Les opérations principales utilisées par le réseau sont :

- addition de matrices : $A + B$
- soustraction de matrices : $A - B$
- produit matriciel : $AB$
- transposée : $A^T$
- produit terme à terme : $A \odot B$
- multiplication par un scalaire : $\alpha A$

### Produit matriciel

Si $A$ est de taille $m \times n$ et $B$ de taille $n \times p$, alors :

$$C = AB \in \mathbb{R}^{m \times p}$$

avec

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

### Transposée

La transposée échange lignes et colonnes :

$$\left(A^T\right)_{ij} = A_{ji}$$

## Propagation avant

Le calcul d’une prédiction suit ces étapes.

### 1) Couche cachée

On calcule d’abord la combinaison linéaire :

$$z_1 = W_1 x + b_1$$

Puis on applique une fonction d’activation sigmoïde :

$$a_1 = \sigma(z_1)$$

La fonction sigmoïde est définie par :

$$\sigma(u) = \frac{1}{1 + e^{-u}}$$

### 2) Couche de sortie

On calcule ensuite :

$$z_2 = W_2 a_1 + b_2$$

Puis on applique le softmax pour obtenir une distribution de probabilité :

$$\hat{y} = \mathrm{softmax}(z_2)$$

### Softmax

Pour un vecteur $z$ de taille 10, le softmax donne :

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

Ce calcul transforme les scores bruts en probabilités qui somment à 1 :

$$\sum_{i=1}^{10} \hat{y}_i = 1$$

## Fonction de perte

Le réseau est entraîné avec l’entropie croisée pour une classification multi-classes.

Si la vraie étiquette est codée en one-hot $y$ et la prédiction est $\hat{y}$, alors :

$$L(y, \hat{y}) = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)$$

Comme $y$ est one-hot, une seule composante vaut 1. Si la vraie classe est $c$, alors :

$$L = -\log(\hat{y}_c)$$

Intuition : plus la probabilité donnée à la bonne classe est grande, plus la perte est petite.

## Rétropropagation

La rétropropagation calcule les gradients de la perte par rapport aux poids et aux biais.

On utilise la combinaison classique **softmax + entropie croisée**, dont le gradient de sortie simplifie fortement.

### Gradient à la sortie

$$\delta_2 = \hat{y} - y$$

Puis :

$$\frac{\partial L}{\partial W_2} = \delta_2 a_1^T$$

$$\frac{\partial L}{\partial b_2} = \delta_2$$

### Gradient vers la couche cachée

On propage l’erreur vers la couche précédente :

$$\frac{\partial L}{\partial a_1} = W_2^T \delta_2$$

La dérivée de la sigmoïde vaut :

$$\sigma'(u) = \sigma(u) \left(1 - \sigma(u)\right)$$

Donc :

$$\delta_1 = \left(W_2^T \delta_2\right) \odot \sigma'(z_1)$$

Puis :

$$\frac{\partial L}{\partial W_1} = \delta_1 x^T$$

$$\frac{\partial L}{\partial b_1} = \delta_1$$

## Mise à jour des paramètres

Le réseau utilise une descente de gradient simple.

Si $\eta$ est le taux d’apprentissage, alors :

$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$

$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

Dans le code, cela correspond à chaque sample d’entraînement.

## Encodage one-hot

Une étiquette MNIST comme `3` est transformée en vecteur one-hot :

$$y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^T$$

Cela permet d’utiliser une formulation vectorielle uniforme pour toutes les classes.

## Prétraitement des images

Les fichiers MNIST bruts contiennent des pixels en niveaux de gris entre `0` et `255`.

Le code normalise chaque pixel avec :

$$p_{norm} = \frac{p}{255}$$

Ainsi chaque entrée du réseau est dans l’intervalle :

$$[0, 1]$$

Dans l’interface SFML, la zone de dessin `280 × 280` est rééchantillonnée vers `28 × 28` en moyennant les blocs correspondants.

## Dimensions utilisées

Pour éviter les erreurs de forme, voici les dimensions utilisées :

- $x \in \mathbb{R}^{784 \times 1}$
- $W_1 \in \mathbb{R}^{128 \times 784}$
- $b_1 \in \mathbb{R}^{128 \times 1}$
- $z_1, a_1 \in \mathbb{R}^{128 \times 1}$
- $W_2 \in \mathbb{R}^{10 \times 128}$
- $b_2 \in \mathbb{R}^{10 \times 1}$
- $z_2, \hat{y} \in \mathbb{R}^{10 \times 1}$

Ces dimensions garantissent que les produits matriciels sont valides :

$$W_1 x : (128 \times 784)(784 \times 1) = 128 \times 1$$

$$W_2 a_1 : (10 \times 128)(128 \times 1) = 10 \times 1$$

## Interprétation de la prédiction

La sortie $\hat{y}$ contient 10 probabilités.

La classe prédite est celle qui a la valeur maximale :

$$\mathrm{classe} = \arg\max_i \hat{y}_i$$

## Sauvegarde des poids

Le programme peut enregistrer les paramètres appris dans un fichier binaire.

Cela permet de :

- entraîner une fois
- réutiliser ensuite les poids pour la prédiction
- ouvrir l’interface SFML sans réentraîner

## Compilation

### Sur Linux

Installer les dépendances (Ubuntu/Debian) :

```bash
sudo apt-get install libsfml-dev g++
```

Compiler :

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp neuralNetwork.cpp drawingApp.cpp trainingGraphWindow.cpp sevenSegmentDigit.cpp -o neuranet -lsfml-graphics -lsfml-window -lsfml-system
```

Lancer :

```bash
./neuranet draw weights.bin
```

### Sur Windows

#### Prérequis

1. Installer **MinGW-w64** (compilateur g++)
   - Télécharger depuis [mingw-w64.org](https://www.mingw-w64.org/)
   - Ajouter le dossier `bin` à la variable `PATH`

2. Installer **SFML 2.6**
   - Télécharger les binaires MinGW depuis [sfml-dev.org](https://www.sfml-dev.org/download/sfml/2.6.0/)
   - Décompresser dans un dossier (ex: `C:\SFML`)

#### Compiler

Remplacer `C:\SFML` par votre chemin d'installation SFML :

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp neuralNetwork.cpp drawingApp.cpp trainingGraphWindow.cpp sevenSegmentDigit.cpp -o neuranet.exe -IC:\SFML\include -LC:\SFML\lib -lsfml-graphics -lsfml-window -lsfml-system
```

#### Lancer sous Windows

Copier les fichiers DLL dans le même dossier que `neuranet.exe` :

```bash
copy C:\SFML\bin\sfml-graphics-2.dll .
copy C:\SFML\bin\sfml-window-2.dll .
copy C:\SFML\bin\sfml-system-2.dll .
```

Puis lancer l'application :

```bash
neuranet.exe draw weights.bin
```

Ou double-cliquer sur `neuranet.exe`.

## Données MNIST attendues

Place les fichiers IDX bruts dans le dossier `data/mnist/` :

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Utilisation

### Entraîner le modèle

Linux :

```bash
./neuranet train data/mnist 3 0.01 weights.bin
```

Windows (dans PowerShell ou CMD) :

```bash
neuranet.exe train data/mnist 3 0.01 weights.bin
```

### Ouvrir l'interface de dessin

Linux :

```bash
./neuranet draw weights.bin
```

Windows :

```bash
neuranet.exe draw weights.bin
```

### Tester le modèle

Linux :

```bash
./neuranet test data/mnist weights.bin
```

Windows :

```bash
neuranet.exe test data/mnist weights.bin
```

Dans la fenêtre SFML :

- maintiens le bouton gauche pour dessiner
- appuie sur `C` pour effacer le canevas
- le canevas de dessin est une grille `28 x 28` (pixels agrandis pour le confort)
- à gauche, les pourcentages de prédiction sont affichés pour les 10 classes (`0` à `9`)

Pendant `train`, une fenêtre de suivi s’ouvre automatiquement pour visualiser l’évolution de la `loss` avec un échantillonnage d’un point sur 100 `forward propagation`.
