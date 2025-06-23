# T-DEV-810

## Introduction

La détection automatisée des pneumonies à partir de radiographies thoraciques constitue un enjeu majeur pour améliorer le diagnostic médical. Ce rapport, réalisé dans le cadre du module **T-DEV-810**, explore différentes approches de classification supervisée pour distinguer trois classes : **Normal**, **Pneumonie Bactérienne** et **Pneumonie Virale**.

L’objectif est d’évaluer et comparer plusieurs modèles classiques — **régression logistique**, **régression linéaire**, **Random Forest**, et **PCA combinée à une régression logistique** — en termes de précision, complexité et capacité de généralisation.

La problématique est la suivante :
**Quelle méthode de classification permet d’obtenir la meilleure précision tout en conservant une certaine interprétabilité et une complexité raisonnable pour le traitement des images médicales ?**

## Table des Matières

---

## 1. Le jeu de données

### Où le trouver ?
Les radiographies se trouvent  dans le dans le dossier [chest_Xray](https://github.com/EpitechMscProPromo2026/T-DEV-810-STG_10/tree/main/chest_Xray):

avec **5216** radiographies de train dont:
  - 1341 radiographies saines
  - 3875 radiographies de pneumonies

et **624** radiographies de tests dont:
  - 234 radiographies saines
  - 390 radiographies de pneumonies

Donc on a environs 88% des images pour l'entraînement et 12% pour les tests

### Comment obtenir les libellés ?

```
.
├── test/
│   ├── NORMAL/
│   │   └── nom_image
│   └── PNEUMONIA/
│       └── nom_image_(virus/bacterie)
└── train/
    ├── NORMAL/
    │   └── nom_image
    └── PNEUMONIA/
        └── nom_image_(virus/bacterie)
```

Les images se trouvant dans les dossiers *NORMAL* sont des radiographies saines, les dossiers *PNEUMONIA* sont des radiographies de pneumonies et le type de la pneumonies se trouve dans le nom de l'image (ex: image_241_virus.jpg)

## Transformations des données pour les Algorithmes

Pour permettre une comparaison entre chaque algorithme on a essayé d'avoir le plus souvent les mêmes traitement d'images à chaque fois, des précisions seront présentes si pour l'algorithme on a fait d'autres transformations.

Les algorithmes demandent d'avoir le même nombre de features[^1] ce qui nous oblige a redimensionné les images dans une taille fixe, pour tous les algorithmes les images sont redimensionné entre 400x400 et 100x100 pixels dont le plus performant étant 128x128.

On a obtenu ce chiffre en testant différentes tailles sur les algorithmes.

Puis les images sont flatten[^2] car le modèle ne prends pas les tableaux à plusieurs dimensions.

> Une image en nuance de gris contient 2 dimensions (position x et y du pixel qui contient la valeur du gris dans le pixel)
> Une image en couleur contient 3 dimensions (position x et y qui contiennent un tableau contenant les valeurs pour les couleurs RGB (Red-Blue-Green))

[^1]: Une feature est un pixel de notre image qu'on souhaite passer dans le modèle
[^2]: Transformations d'un tableau avec X dimensions en un tableau d'une seule dimension 

---

A partir de maintenant on va aborder les modèles tester, les résultats et une observation de nos résultats.
Pour commencer tous les algorithmes en dehors du deep learning proviennent de la librairie `sklearn` qui implémente 
directement les algorithmes facililant la mise en place.

## Algorithme KNearest-Neighbors (KNN)

### Explication

KNN (K-Nearest Neighbors) est un algorithme d'apprentissage supervisé simple qui prédit la classe (ou la valeur) d'un point en se basant sur ses **K** voisins les plus proches dans l'ensemble de données d'entraînement. Il utilise un vote majoritaire parmi les voisins.

### Hyperparamètres

KNN a qu'un seul hyperparamètre, ce dernier étant le nombre de voisin.
Plus K est grand moins il est sensible au bruit, mais perd de la précision sur la différences entre petit détail

### Résultats 

#### Meilleures Résultats

| Paramètres | type classification | Accuracy |
| ---------- | ------------------- | -------- |
| 2          | binaire             | 79%      |
| 12         | ternaire            | 66%      | 

#### Courbes des précisions en fonction du nombre de voisin
##### Binaire
![classification binaire](./knn/classification_binaire.png)

##### Ternaire
![classification ternaire](./knn/classification_trinaire.png)

### Observation 

L'algorithme pourrait être plus intéressant sur des données qui sont plus différentes les une des autres, par exemple une IA qui permet de voir quel
animal est représenté sur l'image car un chien et un chat ont des grandes différences. 
Dans notre cas, les images n'ont pas d'énormes différences visibles, une personne qui n'est pas du domaine n'arriverai pas à dire laquel est une pneumonie ou pas. 

En plus KNN prends énormément de RAM car quand il cherche à prédire une image, il est obligé de mettre en mémoire chaque point de chaque image pour faire les calculs de distance le plus rapidement possible.

On remarque sur les résultats que l'algorithme s'en sort moins bien avec la classification entre virale et bactérienne, car la différence est moins visible qu'entre une personne ayant une pneumonie et une personne qui n'en a pas.

## Détection de la Pneumonie via Random Forest

### Explication

### Hyperparamètre

Pour optimiser mes hyperparamètres j'ai utilisé une fonction de `sklean` se nommant `GridSearchCV` qui prends en paramètres une liste de dictionnaires associant 
le nom de l'hyperparamètre avec une liste de valeur possible. Ainsi cette fonction va exécuter automatiquement le modèle avec les différents hyperparamètres 
et retourne les hyperparamètres qui ont mieux performés.

#### Meilleures Hyperparamètres

Les meilleurs paramètres obtenus sont :

| Hyperparamètre       | Valeur sélectionnée |
| --------------------- | --------------------- |
| `criterion`         | `entropy`           |
| `max_depth`         | `None`              |
| `max_features`      | `sqrt`              |
| `min_samples_split` | `10`                |
| `n_estimators`      | `300`               |

### Évaluation du Modèle

Cette configuration a permis d’atteindre une **précision d’environ 78 %** sur le jeu de test.

### Observation

Le modèle Random Forest constitue une **première approche simple et interprétable** pour classifier des radiographies pulmonaires.
Toutefois, les performances sont limitées (78 % de précision) comparées à d’autres méthodes.
Cela s’explique par l’absence de prise en compte de la structure spatiale des images.
L’utilisation de modèles plus complexes, tels que les **réseaux convolutifs (CNN)**, semble inévitable pour améliorer significativement la performance sur ce type de données.


## Détection de la Pneumonie via Régression Logistique

### Explication

#### Régression logistique

#### PCA

La PCA est appliquée pour réduire la dimensionnalité des vecteurs d’images tout en conservant la majorité de la variance. Différentes configurations ont été testées :

- Nombre de composantes défini par la variance expliquée (n_components=0.95, 0.90, 0.99)
- Nombre fixe de composantes (100, 300)

L’objectif est de trouver un compromis entre richesse des données conservées et complexité du modèle.

### Hyperparamètre

- Mode `multinomial` (classification multi-classe)
- Régularisation de type `L2` (ridge)
- Solveur `saga` ou `lbfgs` selon les cas
- `max_iter` pour le  nombre d'itération max

### Résultat

#### Sans PCA

#### Avec PCA

##### Impact de la taille des images

| ID       | Taille Image | Description                          | Accuracy |
| -------- | ------------ | ------------------------------------ | -------- |
| Baseline | 400x400      | Réglage de base                     | 82%      |
| V1       | 200x200      | Réduction de la taille, plus rapide | 82%      |
| V2       | 128x128      | Taille intermédiaire                | 80%      |
| V3       | 100x100      | Compression agressive                | 85%      |

**Observation :** La réduction agressive à 100x100 améliore légèrement la précision, probablement par effet de régularisation ou réduction du bruit.

##### Influence du nombre de composantes PCA (avec image_size=100x100)

| ID | n_components          | Description                         | Accuracy |
| -- | --------------------- | ----------------------------------- | -------- |
| P0 | 0.95 (variance)       | Baseline                            | 85%      |
| P1 | 0.90 (moins de comp.) | Moins de composantes, plus rapide   | 86%      |
| P2 | 0.99 (plus riche)     | Conserve davantage d’information   | 86%      |
| P3 | 100 (fixe)            | Nombre fixe de composantes          | 84%      |
| P4 | 300 (très riche)     | Risque de bruit ou surapprentissage | 86%      |

##### Réglages du modèle de régression logistique

| ID | Paramètres                 | Description                                    | Accuracy |
| -- | --------------------------- | ---------------------------------------------- | -------- |
| M0 | max_iter=1000               | Baseline                                       | 86%      |
| M1 | max_iter=2000               | Plus d’itérations                            | 84%      |
| M2 | solver='saga'               | Optimisé pour grands jeux de données         | 85%      |
| M3 | penalty='l1', solver='saga' | Régularisation Lasso favorisant la parcimonie | 86%      |
| M4 | C=0.1                       | Régularisation forte, modèle plus simple     | 78%      |
| M5 | C=10.0                      | Faible régularisation, modèle plus flexible  | 84%      |

##### Synthèse des Meilleures Configurations

| Test ID    | Accuracy | Commentaires                                    |
| ---------- | -------- | ----------------------------------------------- |
| V3         | 85%      | Taille image réduite améliore la précision   |
| P1, P2, P4 | 86%      | PCA avec 90%-99% variance conservée optimal    |
| M0, M3     | 86%      | Régression avec L1 et solver 'saga' performant |


### Observation

La **régression logistique multinomiale** constitue une **baseline solide et interprétable** pour des tâches de classification d’images.
Toutefois, ses performances sont limitées dès que la structure spatiale des images devient déterminante, ce qui est le cas pour les radiographies médicales.
Des techniques comme les **réseaux de neurones convolutifs (CNN)** devraient être privilégiées pour améliorer significativement les résultats.

L’utilisation combinée de la réduction de dimension par PCA et d’un modèle de régression logistique permet d’atteindre une précision satisfaisante (~86%) pour la classification de radiographies en trois classes.
Les résultats suggèrent qu’une réduction modérée de la taille des images ainsi qu’un choix judicieux du nombre de composantes PCA améliorent les performances.
La régularisation L1 avec solver ‘saga’ aide à obtenir un modèle plus parcimonieux sans perte de précision notable.

Pour aller plus loin, l’intégration de techniques de Deep Learning, notamment les CNN, serait la voie privilégiée pour exploiter pleinement la nature visuelle des images médicales

## Impact des Paramètres sur la Régression Linéaire

### Hyperparamètres

Le modèle de base repose sur `LinearRegression()` de `scikit-learn`.
La prédiction continue est arrondie avec `np.round()` puis **clipée** dans l’intervalle [0, 2].

| Paramètre           | Valeur                             |
| -------------------- | ---------------------------------- |
| Modèle              | `LinearRegression()`             |
| Taille des images    | `(128, 128)`                     |
| Mode                 | Niveaux de gris, images aplaties   |
| Normalisation        | Pixels divisés par 255            |
| Jeu de test          | 20% des données, stratifié       |
| Arrondi prédictions | `np.round()` + `np.clip(0, 2)` |

**Accuracy obtenue : 73%**

### Résultats

#### Variation de la Taille des Images

| ID | Modification              | Description                              | Résultat (Accuracy) |
| -- | ------------------------- | ---------------------------------------- | -------------------- |
| V1 | `image_size=(64, 64)`   | Taille plus petite, moins de dimensions  | 62%                  |
| V2 | `image_size=(256, 256)` | Taille plus grande, plus d’informations | 71%                  |

**Observation :** Une taille d’image trop réduite nuit à la précision, probablement en raison d’une perte d’information. Une taille supérieure à 128x128 améliore légèrement les performances mais augmente le coût computationnel.

#### Ajustement de l’Intercept

| ID | Modification            | Description               | Résultat (Accuracy) |
| -- | ----------------------- | ------------------------- | -------------------- |
| M1 | `fit_intercept=False` | Ne pas apprendre de biais | 68%                  |

**Observation :** Supprimer le biais (`intercept`) dégrade la précision. Cela montre son importance dans le bon ajustement des prédictions.

### Exemple de Code avec Régression Ridge

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np

# Entraînement du modèle
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Arrondi et clipping des prédictions pour correspondre aux classes 0,1,2
y_pred_rounded = np.clip(np.round(y_pred), 0, 2).astype(int)

# Calcul de l'accuracy
accuracy = np.mean(y_pred_rounded == y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Observation

La régression linéaire simple, bien que peu adaptée de prime abord à la classification, permet ici d’atteindre une précision correcte (73%).
Les expérimentations montrent :

L’importance de la taille des images : trop petite = perte d’information, trop grande = gain modéré + coût accru

Le rôle essentiel du biais (intercept) dans l’apprentissage

Pour améliorer ces résultats, on pourrait explorer :

Des modèles régularisés (Ridge, Lasso)

Des transformations non linéaires

Des approches neuronales ou convolutives

## 7. Conclusion

Résumé des principaux résultats, enseignements, pistes d’amélioration, perspectives futures.
