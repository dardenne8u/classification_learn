# T-DEV-810


# Introduction

La détection automatisée des pneumonies à partir de radiographies thoraciques constitue un enjeu majeur pour améliorer le diagnostic médical. Ce rapport, réalisé dans le cadre du module T-DEV-810, explore différentes approches de classification supervisée pour distinguer trois classes : **Normal**, **Pneumonie Bactérienne** et **Pneumonie Virale**.

L’objectif est d’évaluer et comparer plusieurs modèles — **KNN**, **Random Forest**, **Régression Logistique**, **Régression Linéaire**, **PCA + Régression Logistique**, **Cross-Décomposition**, et **CNN** — en termes de **précision**, **complexité** et **capacité de généralisation**.

La problématique est la suivante :  
**Quelle méthode de classification permet d’obtenir la meilleure précision tout en conservant une certaine interprétabilité et une complexité raisonnable pour le traitement des images médicales ?**

---

# Table des Matières

1. Le jeu de données  
2. Transformations des données pour les algorithmes  
3. K-Nearest Neighbors (KNN)  
4. Random Forest  
5. Régression Logistique (avec/sans PCA)  
6. Régression Linéaire  
7. Cross-Décomposition (PLS)  
8. Convolutional Neural Networks (CNN)  
9. Conclusion

---

# 1. Le jeu de données

Les radiographies se trouvent dans le dossier `chest_Xray`, avec :

- **5216 images pour l’entraînement** : 1341 normales, 3875 pneumonies.  
- **624 images pour les tests** : 234 normales, 390 pneumonies.

Soit environ **88 % pour l’entraînement** et **12 % pour le test**.

Les dossiers `NORMAL` contiennent les radiographies saines, `PNEUMONIA` celles de pneumonies, et les noms d’images (ex. `image_241_virus.jpg`) indiquent le type de pneumonie.

---

# 2. Transformations des données pour les algorithmes

Pour comparer équitablement chaque méthode, nous appliquons le même traitement d’images, sauf mention contraire :

- Redimensionnement à une taille fixe (tests entre 400×400 et 100×100 px, **128×128** le plus performant)  
- Conversion en niveau de gris et aplatissement (**flatten**) pour obtenir un vecteur 1D  
- **Normalisation** des pixels (division par 255) pour certaines méthodes  
- Les pixels sont considérés comme **features individuelles**

---

# 3. K-Nearest Neighbors (KNN)

## Explication  
KNN prédit la classe d’un point d’après le vote majoritaire de ses K plus proches voisins.

## Hyperparamètre  
Nombre de voisins K :
- Plus K est grand → moins sensible au bruit
- Mais moins précis pour les détails fins

## Résultats

| K   | Classification | Accuracy test |
|-----|----------------|----------------|
| 2   | Binaire        | 79 %           |
| 12  | Ternaire       | 66 %           |

## Observation  
- Les différences visuelles entre classes sont subtiles  
- KNN requiert beaucoup de mémoire  
- Moins efficace pour distinguer bactérienne vs virale

---

# 4. Random Forest

## Explication  
Ensemble d’arbres de décision, chaque arbre vote, la forêt tranche à la majorité.

## Optimisation  
Utilisation de `GridSearchCV`.

| Hyperparamètre       | Valeur optimale |
|----------------------|-----------------|
| criterion            | entropy         |
| max_depth            | None            |
| max_features         | sqrt            |
| min_samples_split    | 10              |
| n_estimators         | 300             |

## Performance  
**Accuracy test** : ~78 %

## Observation  
- Simple et interprétable  
- Ignore la structure spatiale de l’image  
- Inférieur aux performances des CNN

---

# 5. Régression Logistique (avec/sans PCA)

## Explication  
Classification multiclasse (multinomial), régularisation **L2**

### Without PCA

- Accuracy (100×100) : 85 %

### With PCA

| ID  | Image     | n_components       | Accuracy |
|-----|-----------|---------------------|----------|
| P1  | 100×100   | 0.90 (variance)     | 86 %     |
| P2  | 100×100   | 0.99 (variance)     | 86 %     |
| P4  | 100×100   | 300 composantes     | 86 %     |

### Optimisation du modèle

| ID  | Paramètres                         | Accuracy |
|-----|------------------------------------|----------|
| M0  | max_iter=1000                      | 86 %     |
| M3  | penalty='l1', solver='saga'        | 86 %     |
| M4  | C=0.1                              | 78 %     |

## Observation  
- Bon compromis précision/interprétabilité (~86 %)  
- PCA réduit la charge et améliore parfois la régularisation  
- Nécessite un modèle spatial (CNN) pour aller plus loin

---

# 6. Régression Linéaire

## Configuration  
- Prédiction continue, arrondie et clipée en [0,2]  
- Images 128×128, niveau de gris, normalisées

## Résultats

| ID  | Modification          | Accuracy |
|-----|-----------------------|----------|
| V1  | 64×64                 | 62 %     |
| V2  | 256×256               | 71 %     |
| M1  | fit_intercept=False   | 68 %     |

## Observation  
- Trop petite taille = perte d’information  
- Intercept nécessaire  
- Modèle simple, améliorable avec Ridge, Lasso

---

# 7. Cross-Décomposition (PLS)

## Explication  
PLS (Partial Least Squares) : réduction de dimension + régression simultanées

## Paramètres testés  
`n_components` = [5, 10, 20, 50]

## Résultats

| ID   | n_components | Accuracy test |
|------|--------------|---------------|
| C5   | 5            | 80 %          |
| C10  | 10           | 84 %          |
| C20  | 20           | 87 %          |
| C50  | 50           | 86 %          |

## Observation  
- Très bon score (~87 %)  
- Méthode hybride : compression + classification  
- Composantes plus explicables que les pixels bruts

---

# 8. Convolutional Neural Networks (CNN)

## Explication  
CNN capturent les motifs spatiaux grâce aux convolutions, pooling, etc.

## Architecture de base

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

baseline = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=[128,128,1]),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
```

## Résultat  
**Accuracy test** : 97 %

## Observation  
- Meilleure performance globale  
- Fort besoin en données et en puissance  
- Transfer learning possible pour améliorer encore

---

# 9. Conclusion

La détection automatique de pneumonies à partir d’imagerie médicale représente un défi à la croisée de la santé et de l’intelligence artificielle. Dans ce projet, nous avons comparé différentes approches de classification supervisée — allant des méthodes linéaires aux réseaux neuronaux convolutifs — dans le but de distinguer les cas **normaux**, de **pneumonies bactériennes** et de **pneumonies virales** à partir de radiographies thoraciques.

Les résultats montrent que les **réseaux de neurones convolutifs (CNN)** se démarquent nettement, atteignant une précision de **97 %**, grâce à leur capacité à exploiter les structures spatiales des images. Toutefois, cette performance a un coût en termes de **complexité computationnelle**, de **temps d’entraînement** et de **besoin en données annotées**.

En parallèle, des méthodes plus classiques comme la **Régression Logistique combinée à la PCA** ou la **Cross-Décomposition (PLS)** atteignent des scores proches de **86–87 %**, avec une interprétabilité appréciable et une complexité bien plus faible. Ces approches se révèlent donc intéressantes dans un cadre de **diagnostic assisté**, où la transparence des décisions est cruciale.

Les algorithmes comme **KNN** ou **Random Forest** offrent une mise en œuvre simple, mais restent limités par leur faible capacité à capter les nuances visuelles fines entre les types de pneumonie. Enfin, la **régression linéaire**, bien qu’inadaptée pour ce type de tâche, a servi de **référence de base** pour évaluer les gains des méthodes plus avancées.

## Perspectives

Pour approfondir ces travaux, plusieurs axes peuvent être explorés :

- L’intégration de **modèles plus profonds** (e.g. ResNet, VGG) via **transfer learning**  
- L’utilisation de **techniques d’augmentation de données** pour améliorer la robustesse des modèles  
- L’application de **méthodes d’explicabilité** comme Grad-CAM ou LIME, afin de mieux comprendre les décisions du modèle  
- Une **validation sur des jeux de données externes** pour évaluer la **généralisation** du modèle

En somme, si les CNN s’imposent comme l’option la plus performante, les méthodes plus légères restent de très bonnes candidates dans des environnements contraints, ou lorsqu’une **interprétabilité fiable** est requise.
