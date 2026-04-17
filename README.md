# NeuraNet

Projet de réseau de neurones en C++ pour reconnaître les chiffres manuscrits du dataset MNIST. J'ai codé une classe `Matrix` from scratch, la propagation avant, la rétropropagation, et une interface SFML pour dessiner un chiffre à la main et voir ce que le réseau devine en temps réel.

Pendant l'entraînement, une fenêtre SFML s'ouvre aussi automatiquement et affiche un graphe en live de la perte (1 point toutes les 100 forward propagations pour pas surcharger).

## Idée générale

Le réseau apprend à associer une image de chiffre manuscrit à une sortie parmi 10 classes, de `0` à `9`.

Une image MNIST est représentée par un vecteur colonne de taille `784`, parce que l'image fait `28 × 28` pixels :

$$x \in \mathbb{R}^{784 \times 1}$$

Le réseau a 2 couches apprenables :

- une couche cachée de taille `128`
- une couche de sortie de taille `10`

Les poids et biais sont :

$$W_1 \in \mathbb{R}^{128 \times 784}, \quad b_1 \in \mathbb{R}^{128 \times 1}$$

$$W_2 \in \mathbb{R}^{10 \times 128}, \quad b_2 \in \mathbb{R}^{10 \times 1}$$

## La classe Matrix

J'utilise ma propre classe `Matrix` à la place de `std::vector` pour stocker les données numériques.

Les opérations dont j'ai eu besoin pour le réseau :

- addition : $A + B$
- soustraction : $A - B$
- produit matriciel : $AB$
- transposée : $A^T$
- produit terme à terme (Hadamard) : $A \odot B$
- multiplication par un scalaire : $\alpha A$

### Produit matriciel

Si $A$ est de taille $m \times n$ et $B$ de taille $n \times p$, alors :

$$C = AB \in \mathbb{R}^{m \times p}$$

avec

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

### Transposée

La transposée échange juste lignes et colonnes :

$$\left(A^T\right)_{ij} = A_{ji}$$

## Propagation avant

Voici comment le réseau calcule une prédiction.

### 1) Couche cachée

On calcule d'abord la combinaison linéaire :

$$z_1 = W_1 x + b_1$$

Puis on passe ça dans une sigmoïde :

$$a_1 = \sigma(z_1)$$

**Pourquoi la sigmoïde ?** Elle écrase les valeurs dans $[0, 1]$ et introduit de la non-linéarité dans le réseau. Sans ça le réseau ne ferait que des combinaisons linéaires et ne pourrait pas apprendre des trucs un peu complexes.

La formule de la sigmoïde :

$$\sigma(u) = \frac{1}{1 + e^{-u}}$$

### 2) Couche de sortie

Ensuite :

$$z_2 = W_2 a_1 + b_2$$

Et on applique le softmax pour obtenir des probabilités :

$$\hat{y} = \mathrm{softmax}(z_2)$$

### Softmax

**Pourquoi le softmax ?** Il transforme les scores bruts en probabilités entre 0 et 1 qui somment à 1. Ça permet d'interpréter directement la sortie comme la confiance du réseau pour chaque chiffre. Les gros scores ressortent encore plus, les petits s'écrasent.

Pour un vecteur $z$ de taille 10 :

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

On a bien :

$$\sum_{i=1}^{10} \hat{y}_i = 1$$

## Fonction de perte

J'utilise l'entropie croisée pour la classification multi-classes.

Si $y$ est le label encodé en one-hot et $\hat{y}$ la prédiction :

$$L(y, \hat{y}) = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)$$

Comme $y$ est one-hot, un seul terme vaut 1. Si la vraie classe est $c$ :

$$L = -\log(\hat{y}_c)$$

En gros : plus le réseau est confiant sur la bonne classe, plus la perte est petite. Logique.

## Rétropropagation

C'est la partie la plus math-lourde. On calcule les gradients de la perte par rapport aux poids et biais, pour savoir dans quelle direction les ajuster.

### C'est quoi un gradient ?

Un gradient mesure comment la perte réagit quand on modifie légèrement un paramètre. Si le gradient est positif sur un poids, ça veut dire que l'augmenter fait monter la perte, donc il faut le diminuer. Et inversement. En suivant ces gradients à l'envers, on améliore petit à petit le modèle.

### Gradient à la sortie

La combinaison softmax + entropie croisée a un gradient de sortie super simple :

$$\delta_2 = \hat{y} - y$$

Puis :

$$\frac{\partial L}{\partial W_2} = \delta_2 a_1^T$$

$$\frac{\partial L}{\partial b_2} = \delta_2$$

### Gradient vers la couche cachée

On propage l'erreur en arrière :

$$\frac{\partial L}{\partial a_1} = W_2^T \delta_2$$

La dérivée de la sigmoïde vaut :

$$\sigma'(u) = \sigma(u) \left(1 - \sigma(u)\right)$$

C'est pratique parce qu'on l'a déjà calculée lors du forward pass. Donc :

$$\delta_1 = \left(W_2^T \delta_2\right) \odot \sigma'(z_1)$$

Et enfin :

$$\frac{\partial L}{\partial W_1} = \delta_1 x^T$$

$$\frac{\partial L}{\partial b_1} = \delta_1$$

## Mise à jour des paramètres

Descente de gradient classique. Avec $\eta$ le taux d'apprentissage :

$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$

$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

Dans le code ça se fait pour chaque sample d'entraînement (stochastic gradient descent).

## Encodage one-hot

Un label MNIST comme `3` est converti en vecteur one-hot :

$$y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^T$$

Ça permet d'avoir une formulation vectorielle uniforme pour toutes les classes.

## Prétraitement des images

Les fichiers MNIST bruts ont des pixels entre `0` et `255`. Je normalise simplement :

$$p_{norm} = \frac{p}{255}$$

Donc toutes les entrées du réseau sont dans $[0, 1]$.

## Résumé des dimensions

Pour s'y retrouver (je me suis trompé plusieurs fois au début) :

- $x \in \mathbb{R}^{784 \times 1}$
- $W_1 \in \mathbb{R}^{128 \times 784}$, $b_1 \in \mathbb{R}^{128 \times 1}$
- $z_1, a_1 \in \mathbb{R}^{128 \times 1}$
- $W_2 \in \mathbb{R}^{10 \times 128}$, $b_2 \in \mathbb{R}^{10 \times 1}$
- $z_2, \hat{y} \in \mathbb{R}^{10 \times 1}$

Vérification des produits :

$$W_1 x : (128 \times 784)(784 \times 1) = 128 \times 1 \checkmark$$

$$W_2 a_1 : (10 \times 128)(128 \times 1) = 10 \times 1 \checkmark$$

## Interprétation de la prédiction

$\hat{y}$ contient 10 probabilités. La classe prédite c'est simplement celle avec la valeur max :

$$\mathrm{classe} = \arg\max_i \hat{y}_i$$

## Sauvegarde des poids

Le programme peut enregistrer les poids dans un fichier binaire. C'est utile parce que l'entraînement prend du temps, donc on peut entraîner une fois, sauvegarder, et réutiliser les poids plus tard pour juste tester ou ouvrir l'interface de dessin.

## Explications détaillées des formules

Cette section relie chaque formule aux fonctions du code (`Matrix`, `forward`, `trainSample`, etc.).

### 1) Convention de représentation

- Une image MNIST est vectorisée en colonne : $x \in \mathbb{R}^{784 \times 1}$
- Une colonne = un échantillon, une ligne = une feature

Ce choix permet d'écrire tout le forward/backward sous forme de produits matriciels compacts.

### 2) Pourquoi ces dimensions ?

Le réseau a `784` entrées, `128` neurones cachés, `10` sorties.

Chaque neurone d'une couche reçoit toutes les activations de la couche précédente, d'où :

- `128` neurones × `784` entrées → $W_1$ est `128×784`
- `10` neurones × `128` activations → $W_2$ est `10×128`

### 3) Forward pass détaillé

#### Couche cachée

$$z_1 = W_1 x + b_1 \quad \rightarrow \quad (128 \times 784)(784 \times 1) + (128 \times 1) = 128 \times 1$$

$$a_1 = \sigma(z_1) \quad \text{avec} \quad \sigma(u)=\frac{1}{1+e^{-u}}$$

Avantages de la sigmoïde ici : non-linéarité, valeurs bornées dans $[0,1]$, dérivée simple $\sigma'(u)=\sigma(u)(1-\sigma(u))$.

#### Couche de sortie

$$z_2 = W_2 a_1 + b_2 \quad \rightarrow \quad (10 \times 128)(128 \times 1) + (10 \times 1) = 10 \times 1$$

$$\hat{y} = \text{softmax}(z_2) \quad \text{avec} \quad \hat{y}_i = \frac{e^{z_{2,i}}}{\sum_{j=1}^{10}e^{z_{2,j}}}$$

### 4) Perte et intuition

$$L=-\log(\hat{y}_c)$$

Si le modèle donne une forte proba à la bonne classe → perte faible. Si il se plante → perte élevée.

### 5) Backward pass justifié

Pour softmax + cross-entropy, le gradient se simplifie joliment :

$$\delta_2 = \hat{y} - y$$

$$\frac{\partial L}{\partial W_2} = \delta_2 a_1^T \quad \rightarrow \quad (10 \times 1)(1 \times 128) = 10 \times 128 \checkmark$$

Propagation vers la couche cachée (règle de chaîne avec la dérivée sigmoïde) :

$$\delta_1 = (W_2^T\delta_2) \odot \sigma'(z_1)$$

$$\frac{\partial L}{\partial W_1}=\delta_1 x^T, \quad \frac{\partial L}{\partial b_1}=\delta_1$$

C'est ce qui est implémenté dans `trainSample`.

### 6) Mise à jour

$$W \leftarrow W - \eta\frac{\partial L}{\partial W}, \quad b \leftarrow b - \eta\frac{\partial L}{\partial b}$$

On avance dans la direction opposée au gradient pour diminuer la perte localement.

### 7) Vue d'ensemble de l'algo

#### Entraînement

1. Charger MNIST (`loadMnistDataset`) et normaliser : $p/255$
2. Initialiser poids (petites valeurs aléatoires) et biais (zéros)
3. Pour chaque époque :
   - mélanger les samples
   - pour chaque sample : forward → calcul de la perte → backward → update
4. Calculer la précision sur le jeu de test (`evaluate`)
5. Sauvegarder les poids (`saveWeights`)

#### Inférence (mode dessin)

1. Lire le canevas `28×28`
2. Convertir en vecteur `784×1`
3. Appeler `predict` (forward + softmax)
4. Afficher les probas et la classe prédite : $\arg\max_i \hat{y}_i$

### 8) Complexité calculatoire

Par sample, les opérations les plus coûteuses sont :

- $W_1x$ : $\mathcal{O}(128 \times 784)$
- $W_2a_1$ : $\mathcal{O}(10 \times 128)$

La rétropropagation est du même ordre. La complexité est donc dominée par la taille de la couche cachée et la dimension d'entrée.

## Compilation

### Linux

Installer les dépendances (Ubuntu/Debian) :

```bash
sudo apt-get install g++ libsfml-dev
```

Compiler :

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp neuralNetwork.cpp drawingApp.cpp trainingGraphWindow.cpp sevenSegmentDigit.cpp -o neuranet -lsfml-graphics -lsfml-window -lsfml-system
```

Lancer :

```bash
./neuranet draw weights.bin
```

### Windows

#### Étape 1 : ouvrir MSYS2 UCRT64

Attention à utiliser le shell **UCRT64** de MSYS2, pas MinGW64.

#### Étape 2 : installer les outils

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-sfml pkg-config
```

Appuyer sur Entrée deux fois pour confirmer.

#### Étape 3 : compiler

Se déplacer dans le dossier du projet :

```bash
cd /c/NeuraNet
```

Puis compiler :

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp neuralNetwork.cpp drawingApp.cpp trainingGraphWindow.cpp sevenSegmentDigit.cpp -o neuranet.exe $(pkg-config --cflags --libs sfml-graphics sfml-window sfml-system)
```

#### Étape 4 : lancer

```bash
./neuranet.exe train data/mnist 3 0.01 weights.bin
```

> Note : le même code source compile sur Linux et Windows UCRT64 sans aucune modification.

## Données MNIST

Les fichiers IDX bruts doivent être dans `data/mnist/` :

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Utilisation

### Entraîner le modèle

```bash
# Linux
./neuranet train data/mnist 3 0.01 weights.bin

# Windows
./neuranet.exe train data/mnist 3 0.01 weights.bin
```

### Ouvrir l'interface de dessin

```bash
# Linux
./neuranet draw weights.bin

# Windows
./neuranet.exe draw weights.bin
```

### Tester le modèle

```bash
# Linux
./neuranet test data/mnist weights.bin

# Windows
./neuranet.exe test data/mnist weights.bin
```

Dans la fenêtre SFML :

- maintenir clic gauche pour dessiner
- appuyer sur `C` pour effacer le canevas
- le canevas est une grille `28×28`
- à gauche s'affichent les pourcentages pour chaque classe de `0` à `9`

Pendant `train`, la fenêtre de suivi de la loss s'ouvre automatiquement.
