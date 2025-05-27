# Rapport de classification d'images : Détection de la pneumonie via Random Forest

## 1. Introduction

Ce projet a pour but de classifier des radiographies thoraciques en deux catégories : **normal**, **pneumonie bacterien** et **pneumonie virus** .  
Nous utilisons un modèle Random Forest sur des images médicales prétraitées pour évaluer les performances d'une approche classique non convolutive.

## 2. Exploration des données

Le dataset utilisé est issu d’un répertoire contenant trois sous-ensembles :

- Entraînement (`train`)
- Validation (`val`)
- Test (`test`)

Les images sont réparties dans deux classes : `NORMAL`, `BACTERIA` et `VIRUS`.  
Une visualisation RGB des images ainsi que leurs histogrammes de couleurs permet d’avoir un aperçu de la distribution des canaux.

## 3. Prétraitement des données

Les étapes de prétraitement sont les suivantes :

- Lecture et redimensionnement des images en 128x128 pixels
- Conversion BGR → RGB
- Encodage des labels en 0 et 1
- Séparation en X (images) et y (étiquettes)

## 4. Extraction des caractéristiques

Les images sont transformées en vecteurs de caractéristiques afin de pouvoir être utilisées avec un modèle de type Random Forest.  
On peut, par exemple, aplatir l’image ou extraire des descripteurs statistiques simples.

## 5. Modélisation avec Random Forest

Le modèle Random Forest est entraîné sur les données prétraitées.

- Implémentation via `sklearn.ensemble.RandomForestClassifier`
- Entraînement sur le jeu `train`, validation sur `val`, test final sur `test`

#### Paramètres optimaux sélectionnés via GridSearchCV

| Hyperparamètre        | Valeur sélectionnée |
|------------------------|---------------------|
| `criterion`            | `entropy`           |
| `max_depth`            | `None`              |
| `max_features`         | `sqrt`              |
| `min_samples_split`    | `10`                |
| `n_estimators`         | `300`               |


- Qui donne une précision d'environ 78 %


## 6. Évaluation du modèle

Les performances du modèle sont évaluées à l’aide de :

- Matrice de confusion
- Rapport de classification : précision, rappel, F1-score
- Analyse des erreurs (exemples d’images mal classées si disponible)

## 7. Conclusion

Le modèle Random Forest a permis de poser une première base de classification sur des images médicales.  
Cependant, cette approche est limitée par sa simplicité : une méthode de Deep Learning (CNN) serait plus adaptée à la nature visuelle des données.

