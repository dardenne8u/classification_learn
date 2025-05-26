# 🔬 Expérimentation – Impact des Paramètres sur la Précision (Accuracy)

Ce document décrit une série d’expérimentations pour analyser l’impact de différents réglages (prétraitement des images, paramètres PCA, configuration du modèle) sur la **précision d’un modèle PCA + Régression Logistique** destiné à détecter **trois classes** : Normal, Pneumonie Bactérienne, Pneumonie Virale.

---

## ⚙️ Réglages de Base (Baseline)

| Paramètre           | Valeur                                |
| -------------------- | ------------------------------------- |
| Modèle              | `LogisticRegression(max_iter=1000)` |
| Réduction dimension | `PCA(n_components=0.95)`            |
| Taille des images    | `(400, 400)`                        |
| Format               | `Grayscale`(images aplaties)        |
| Normalisation        | Pixels entre 0 et 1                   |
| Split                | 80% train / 20% test, stratifié      |

**Accuracy de base :** 82%

---

## 🧪 Batterie de Tests

### 🔁 1. Variation de la taille d’image

| ID | Modification              | Description                                    | Résultat (Accuracy) |
| -- | ------------------------- | ---------------------------------------------- | -------------------- |
| V1 | `image_size=(200, 200)` | Moins de pixels, traitement plus rapide        | *82%*              |
| V2 | `image_size=(128, 128)` | Test avec 128 pixel                            | 80%                  |
| V3 | `image_size=(100, 100)` | Compression agressive, possible perte d’infos | 85%                  |

---

### 📉 2. PCA – Réduction de dimension

Test Reduction PCA avec `image_size=(100, 100)`.

| ID | Modification          | Description                          | Résultat (Accuracy) |
| -- | --------------------- | ------------------------------------ | -------------------- |
| P0 | `n_components=0.95` | n_component base                     | 85%                  |
| P1 | `n_components=0.90` | Moins de composantes, plus rapide    | 86%                  |
| P2 | `n_components=0.99` | Préserve davantage de variance      | 86%                  |
| P3 | `n_components=100`  | Nombre fixe de composantes           | 84%                  |
| P4 | `n_components=300`  | Très riche, risque de bruit inutile | 86%                  |

---

### ⚙️ 3. Modifications du modèle de régression logistique

| ID | Modification      | Description                                                                                | Résultat (Accuracy) |
| -- | ----------------- | ------------------------------------------------------------------------------------------ | -------------------- |
| M0 | `max_iter=1000` |                                                                                            | 86%                  |
| M1 | `max_iter=2000` | Plus d’itérations pour convergence                                                       | 84%                  |
| M2 | `solver='saga'` | Optimisé pour les grands jeux de données                                                 | 85%                  |
| M3 | `penalty='l1'`  | Lasso : favorise des poids nuls (sparse model)<br />`max_iter=1000` + `solver='saga'` | *86%*              |
| M4 | `C=0.1`         | Régularisation forte                                                                      | 78%                  |
| M5 | `C=10.0`        | Faible régularisation, plus de flexibilité                                               | *84 %*             |

---

## 📊 Résumé des Expériences

| Test ID | Accuracy | Observations                                                    |
| ------- | -------- | --------------------------------------------------------------- |
| V1      | 86%      | Réduction de taille → plus rapide, attention à la précision |
| P2      | 86%      | PCA plus riche → utile si perte d’info                        |
| M3      | 86%      | Modèle plus sparse, mais perte possible de performance         |

---

## 🛠️ Code type pour une expérimentation

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('pca', PCA(n_components=0.99)),
    ('clf', LogisticRegression(
        solver='saga',
        penalty='l1',
        C=0.5,
        max_iter=2000
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
