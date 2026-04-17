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

**Utilité :** La sigmoïde aplati les valeurs en sortie de la couche cachée dans l'intervalle $[0, 1]$. Cela introduit de la non-linéarité dans le réseau, permettant d'apprendre des relations complexes entre les entrées et les sorties. Sans fonction d'activation, le réseau serait juste une combinaison linéaire et ne pourrait pas résoudre des problèmes non-linéaires.

La fonction sigmoïde est définie par :

$$\sigma(u) = \frac{1}{1 + e^{-u}}$$

Elle transforme les valeurs de $z_1$ pour obtenir :

$$a_1 = \sigma(z_1)$$

### 2) Couche de sortie

On calcule ensuite :

$$z_2 = W_2 a_1 + b_2$$

Puis on applique le softmax pour obtenir une distribution de probabilité :

$$\hat{y} = \mathrm{softmax}(z_2)$$

### Softmax

**Utilité :** Le softmax convertit les valeurs de sortie du réseau en une distribution de probabilité. Cela permet d'interpréter chaque composante comme la confiance du réseau pour chaque classe de 0 à 9. Les scores élevés sont amplifiés tandis que les scores faibles sont atténués, rendant la prédiction plus discriminante.

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

### Qu'est-ce qu'un gradient ?

Un gradient est une mesure de la façon dont une fonction change lorsque ses entrées changent. Dans le contexte de l'apprentissage automatique, il nous indique dans quelle direction nous devons ajuster nos poids pour réduire l'erreur de prédiction. En d'autres termes, il nous aide à savoir comment améliorer notre modèle.

Lorsque nous calculons le gradient, nous cherchons à comprendre comment une petite modification des poids ou des biais affecte la perte. Si le gradient est positif, cela signifie que nous devons diminuer le poids pour réduire la perte. Si le gradient est négatif, nous devons l'augmenter. En suivant ces gradients, nous pouvons ajuster nos paramètres pour améliorer les performances de notre modèle.

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

## Justification détaillée des équations et matrices

Cette section relie explicitement chaque formule aux objets manipulés dans le code (`Matrix`, `forward`, `trainSample`, etc.).

### 1) Convention de représentation

- Une image MNIST est vectorisée en colonne :

$$x \in \mathbb{R}^{784 \times 1}$$

- Une colonne = un échantillon.
- Une ligne = une caractéristique (un pixel pour l’entrée, un neurone pour les couches internes/sortie).

Ce choix permet d’écrire toutes les étapes d’un passage avant et arrière sous forme de produits matriciels compacts.

### 2) Pourquoi ces dimensions de matrices ?

Le réseau a `784` entrées, `128` neurones cachés, `10` sorties.

- Couche cachée :

$$W_1 \in \mathbb{R}^{128 \times 784}, \quad b_1 \in \mathbb{R}^{128 \times 1}$$

- Couche sortie :

$$W_2 \in \mathbb{R}^{10 \times 128}, \quad b_2 \in \mathbb{R}^{10 \times 1}$$

Justification : chaque neurone d’une couche reçoit _toutes_ les activations de la couche précédente.

- `128` neurones cachés × `784` entrées $\Rightarrow$ `W1` est `128 x 784`.
- `10` neurones de sortie × `128` activations cachées $\Rightarrow$ `W2` est `10 x 128`.

### 3) Propagation avant, étape par étape

#### Couche cachée

$$z_1 = W_1 x + b_1$$

Dimensions :

$$ (128 \times 784)(784 \times 1) + (128 \times 1) = 128 \times 1 $$

Puis activation :

$$a_1 = \sigma(z_1), \quad \sigma(u)=\frac{1}{1+e^{-u}}$$

Pourquoi sigmoïde ici ?

- introduit de la non-linéarité,
- borne les activations dans $[0,1]$,
- garde une dérivée simple :

$$\sigma'(u)=\sigma(u)(1-\sigma(u))$$

#### Couche de sortie

$$z_2 = W_2 a_1 + b_2$$

Dimensions :

$$ (10 \times 128)(128 \times 1) + (10 \times 1) = 10 \times 1 $$

Puis :

$$\hat{y} = \text{softmax}(z_2)$$

avec

$$\hat{y}_i = \frac{e^{z_{2,i}}}{\sum_{j=1}^{10}e^{z_{2,j}}}$$

Pourquoi softmax ?

- transforme des scores réels en probabilités,
- force $\sum_i \hat{y}_i = 1$,
- rend la sortie directement interprétable pour une classification multi-classes.

### 4) Fonction de perte et justification

Avec un label one-hot $y$ :

$$L(y,\hat{y}) = -\sum_{i=1}^{10} y_i\log(\hat{y}_i)$$

Comme $y$ contient un seul `1` (classe correcte $c$), on obtient :

$$L=-\log(\hat{y}_c)$$

Interprétation :

- si le modèle donne une forte probabilité à la bonne classe, la perte devient faible,
- s’il donne une faible probabilité à la bonne classe, la perte devient grande.

### 5) Rétropropagation justifiée

Pour le couple `softmax + cross-entropy`, le gradient en sortie se simplifie en :

$$\delta_2 = \hat{y} - y$$

Puis :

$$\frac{\partial L}{\partial W_2} = \delta_2 a_1^T, \quad \frac{\partial L}{\partial b_2} = \delta_2$$

Dimensions :

$$ (10 \times 1)(1 \times 128)=10 \times 128 $$

ce qui correspond exactement à la taille de $W_2$.

Propagation vers la couche cachée :

$$\frac{\partial L}{\partial a_1} = W_2^T\delta_2$$

Puis application de la règle de chaîne avec la dérivée sigmoïde :

$$\delta_1 = (W_2^T\delta_2) \odot \sigma'(z_1)$$

et enfin :

$$\frac{\partial L}{\partial W_1}=\delta_1 x^T, \quad \frac{\partial L}{\partial b_1}=\delta_1$$

Cette forme est celle implémentée dans `trainSample`.

### 6) Mise à jour des paramètres

Descente de gradient :

$$W \leftarrow W - \eta\frac{\partial L}{\partial W}, \quad b \leftarrow b - \eta\frac{\partial L}{\partial b}$$

où $\eta$ est le taux d’apprentissage.

Justification : on avance dans la direction opposée au gradient pour diminuer localement la perte.

### 7) Algorithme global (vue d’ensemble)

#### Entraînement

1. Charger MNIST (`loadMnistDataset`) et normaliser les pixels : $p/255$.
2. Initialiser poids et biais (petites valeurs aléatoires + biais nuls).
3. Pour chaque époque :
   - mélanger l’ordre des échantillons,
   - pour chaque échantillon :
     - extraire la colonne image,
     - propagation avant,
     - calcul de la perte,
     - rétropropagation,
     - mise à jour des paramètres.
4. Calculer la précision via `evaluate`.
5. Sauvegarder les poids (`saveWeights`).

#### Inférence (mode dessin)

1. Lire le canevas `28 x 28`.
2. Le convertir en vecteur `784 x 1`.
3. Exécuter `predict` (forward + softmax).
4. Afficher les probabilités et la classe :

$$\arg\max_i \hat{y}_i$$

### 8) Coût de calcul (ordre de grandeur)

Par échantillon, les coûts dominants sont les deux produits matriciels :

- $W_1x$ : $\mathcal{O}(128\cdot784)$
- $W_2a_1$ : $\mathcal{O}(10\cdot128)$

et, en rétropropagation, des coûts du même ordre.

Donc la complexité est principalement pilotée par la taille de la couche cachée et la dimension d’entrée.

## Compilation

Pour Windows, il doit être compilé dans MSYS2 UCRT64 avec SFML installé via `pacman`.

### Sur Linux

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

### Sur Windows

#### Étape 1 : ouvrir MSYS2 UCRT64

Utilisez le shell UCRT64 de MSYS2, pas le shell MinGW64 classique.

#### Étape 2 : installer les outils

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-sfml pkg-config
```

- appuyer sur entree
- entree 'y'

#### Étape 3 : compiler

##### Deplacer vous dans le dossier du projet

par exemple:

```bash
cd /c/NeuraNet
```

##### Lancer :

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp neuralNetwork.cpp drawingApp.cpp trainingGraphWindow.cpp sevenSegmentDigit.cpp -o neuranet.exe $(pkg-config --cflags --libs sfml-graphics sfml-window sfml-system)
```

#### Étape 4 : lancer

```bash
./neuranet.exe train data/mnist 3 0.01 weights.bin
```

### Notes

- Le même code source est compilé sur Linux et sur UCRT64 avec les mêmes fichiers.

## Données MNIST attendues

Les fichiers IDX bruts avec les images sont dans le dossier `data/mnist/` :

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
./neuranet.exe train data/mnist 3 0.01 weights.bin
```

### Ouvrir l'interface de dessin

Linux :

```bash
./neuranet draw weights.bin
```

Windows :

```bash
./neuranet.exe draw weights.bin
```

### Tester le modèle

Linux :

```bash
./neuranet test data/mnist weights.bin
```

Windows :

```bash
./neuranet.exe test data/mnist weights.bin
```

Dans la fenêtre SFML :

- maintiens le bouton gauche pour dessiner
- appuie sur `C` pour effacer le canevas
- le canevas de dessin est une grille `28 x 28`
- à gauche, les pourcentages de prédiction sont affichés pour les 10 classes (`0` à `9`)

Pendant `train`, une fenêtre de suivi s’ouvre automatiquement pour visualiser l’évolution de la `loss` avec un échantillonnage d’un point sur 100 `forward propagation`.
