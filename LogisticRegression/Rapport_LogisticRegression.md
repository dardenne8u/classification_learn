# Rapport de classification d'images : Détection de la pneumonie via Régression Logistique

## 1. Introduction

Ce projet a pour objectif de classifier des radiographies thoraciques en trois catégories : **normal**, **pneumonie bactérienne** et **pneumonie virale**.  
Nous utilisons ici un modèle de **régression logistique multinomiale** sur des images médicales prétraitées afin d’évaluer les performances d’une approche classique linéaire.

## 2. Exploration des données

Le dataset est organisé en trois sous-ensembles :

- Entraînement (`train`)
- Validation (`val`)
- Test (`test`)

Les images sont réparties en trois classes : `NORMAL`, `BACTERIA` et `VIRUS`.  
Une visualisation RGB des images ainsi que l’analyse des histogrammes de couleur a été effectuée pour mieux comprendre la distribution des canaux et des intensités.

## 3. Prétraitement des données

Les étapes de prétraitement comprennent :

- Redimensionnement des images à 128x128 pixels  
- Conversion du format BGR vers RGB  
- Encodage des labels en entiers (`0`, `1`, `2`)  
- Séparation des données en `X` (images) et `y` (étiquettes)

## 4. Extraction des caractéristiques

Les images étant en format matriciel, elles ont été transformées en vecteurs de caractéristiques afin d’être compatibles avec le modèle de régression logistique.

- Aplatissement des images (transformation en vecteurs 1D)  
- Optionnel : normalisation des intensités (mise à l’échelle des pixels entre 0 et 1)

## 5. Modélisation avec Régression Logistique

Le modèle de régression logistique est entraîné pour prédire l’une des trois classes.

- Implémentation avec `sklearn.linear_model.LogisticRegression`  
- Utilisation du mode `multinomial` avec une régularisation (`l2`) et un solveur adapté (`saga`, `lbfgs`, etc.)  
- Entraînement sur le jeu `train`, validation sur `val`, évaluation finale sur `test`

## 6. Évaluation du modèle

Les performances sont évaluées avec les métriques classiques de classification :

- **Matrice de confusion**  
- **Rapport de classification** : précision, rappel, F1-score pour chaque classe  
- Analyse des erreurs : visualisation d'exemples d’images mal classées (si disponible)

## 7. Conclusion

La régression logistique permet de poser une baseline simple pour la classification d’images médicales.  
Toutefois, en raison de sa nature linéaire et de la complexité visuelle des images, cette approche montre des limites.  
Pour améliorer les résultats, des techniques plus avancées telles que les réseaux de neurones convolutifs (CNN) sont recommandées.
