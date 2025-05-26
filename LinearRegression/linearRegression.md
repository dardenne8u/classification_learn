
# 🔬 Expérimentation – Impact des Paramètres sur la Précision (Accuracy)

Ce document décrit une série d’expériences visant à analyser l’impact de diverses **modifications** (prétraitement, taille d’image, réglages du modèle, etc.) sur la **précision du modèle de régression linéaire** appliqué à la classification de radiographies thoraciques.

---

## ⚙️ Réglages de Base (Baseline)

| Paramètre           | Valeur                           |
| -------------------- | -------------------------------- |
| Modèle              | `LinearRegression()`           |
| Taille des images    | `(128, 128)`                   |
| Mode                 | `Grayscale`, aplatie           |
| Normalisation        | Pixels divisés par 255          |
| Jeu de test          | 20% des données, stratifié     |
| Arrondi prédictions | `np.round()`+`np.clip(0, 2)` |

**Accuracy de base :** `73%`

---

## 🧪 Batterie de Tests

### 🔁 1. Variation du prétraitement

| ID | Modification              | Description                              | Résultat (Accuracy) |
| -- | ------------------------- | ---------------------------------------- | -------------------- |
| V1 | `image_size=(64, 64)`   | Taille plus petite, moins de dimensions  | 62%                  |
| V2 | `image_size=(256, 256)` | Taille plus grande, plus d’informations | 71%                  |

---

### ⚙️ 2. Modifications des paramètres ou du modèle

Même si `LinearRegression` a peu d’hyperparamètres directs, certaines alternatives peuvent être testées.

| ID | Modification            | Description                           | Résultat (Accuracy) |
| -- | ----------------------- | ------------------------------------- | -------------------- |
| M1 | `fit_intercept=False` | Ne pas apprendre de biais (intercept) | 68%                  |


## 🛠️ Code type pour une expérimentation

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_rounded = np.clip(np.round(y_pred), 0, 2).astype(int)
accuracy = np.mean(y_pred_rounded == y_test)
print(f"Accuracy: {accuracy:.2f}")

```
