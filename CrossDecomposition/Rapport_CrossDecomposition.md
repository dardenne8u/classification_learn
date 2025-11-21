# Rapport de classification d’images : Détection de la pneumonie par régression PLS (Cross Decomposition)

## 1. Introduction

Ce projet vise à classifier des radiographies pulmonaires en trois catégories distinctes : **normal**, **pneumonie bactérienne** et **pneumonie virale**.
Pour cela, nous avons recours à un modèle de **régression par les moindres carrés partiels (PLSRegression)**, issu de la famille des méthodes de décomposition croisée, afin d’évaluer son efficacité dans le cadre d’une classification multi-classes.

## 2. Exploration des données

Le jeu de données est structuré selon trois sous-ensembles :

* Données d'entraînement (`train`)
* Données de validation (`val`)
* Données de test (`test`)

Les images sont classées dans trois catégories : `NORMAL`, `BACTERIA` et `VIRUS`.
Une exploration visuelle en RGB et une analyse des histogrammes de couleurs ont été réalisées afin de mieux cerner la répartition des intensités et des canaux.

## 3. Prétraitement des données

Les opérations de prétraitement comprennent :

* Redimensionnement des images à 128x128 pixels
* Conversion du format BGR en RGB
* Encodage des étiquettes sous forme entière (`0`, `1`, `2`)
* Séparation entre les données `X` (images) et `y` (catégories)

## 4. Extraction des caractéristiques

Les images sont converties en vecteurs unidimensionnels pour être compatibles avec la méthode PLS :

* Aplatissement des matrices d’images
* Normalisation des intensités (optionnelle) sur l’intervalle \[0, 1]

## 5. Modélisation avec PLSRegression

Le modèle PLSRegression apprend à projeter les images dans un espace latent corrélé aux étiquettes cibles :

* Utilisation de `sklearn.cross_decomposition.PLSRegression`
* Transformation des labels en encodage one-hot pour un apprentissage de type régressif
* Prédiction par identification de la sortie maximale
* Entraînement sur l’ensemble `train`, validation sur `val`, évaluation finale sur `test`

## 6. Évaluation du modèle

L’évaluation s’appuie sur des métriques classiques de classification :

* **Matrice de confusion**
* **Rapport de classification** comprenant précision, rappel et F1-score
* Étude des erreurs à travers des exemples d’images mal classées

## 7. Conclusion

La méthode PLSRegression, bien qu’initialement conçue pour des tâches de régression, montre des performances satisfaisantes en classification lorsqu’on encode les étiquettes.
Elle constitue une baseline pertinente grâce à sa capacité à extraire des composantes latentes partagées entre les images et les classes.
Cependant, l’absence de mécanismes non linéaires limite sa capacité à modéliser des frontières de décision complexes.
Des approches plus avancées comme les réseaux de neurones convolutifs (CNN) restent mieux adaptées à ce type de données.
