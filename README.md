# 🧠 Implémentation d’un Réseau Récurrent à Horloge (CW-RNN) avec TensorFlow

Ce projet implémente une version spécialisée d’un **Réseau Récurrent à Horloge (Clockwork Recurrent Neural Network – CW-RNN)** en utilisant **TensorFlow** et **Keras**.

Le **CW-RNN** est une extension des réseaux récurrents standards, où certaines unités sont mises à jour à des intervalles de temps réguliers (appelés *périodes d’horloge*), tandis que les autres restent figées, ce qui permet de mieux capturer les dépendances à long terme tout en réduisant les coûts de calcul.

---

## 🧩 Composants du projet

### 🔁 1. Cellule CW-RNN (`CWRNNCell`)
- Hérite de `SimpleRNNCell` de Keras.
- Ajoute une **logique de mise à jour périodique** basée sur un compteur interne (`step`).
- À chaque appel, elle vérifie si le pas de temps actuel est un multiple de la période définie.
- Si oui → met à jour l’état ; sinon → conserve l’état précédent.
- Cela permet de simuler différentes fréquences internes de traitement de l’information.

---

### 🧠 2. Modèle CW-RNN (`CWRNN`)
- Construit comme une séquence de plusieurs `CWRNNCell`.
- Initialise et gère les états internes.
- Applique les cellules séquentiellement aux données d’entrée selon leur logique d’horloge.

---

### 📊 3. Classificateur CW-RNN (`CWRNNClassifier`)
- Enveloppe le modèle CW-RNN dans une **interface compatible `scikit-learn`**.
- Permet l’intégration facile avec des outils comme `GridSearchCV`, `RandomizedSearchCV`.
- Méthodes disponibles :
  - `fit` : entraînement du modèle
  - `predict` : prédiction
  - `score` : évaluation
  - `save_model` / `load_model` : sauvegarde et chargement du modèle

---

## 🚀 Objectifs

- Réduire la complexité temporelle des RNNs classiques
- Améliorer la capture des **dépendances à long terme**
- Intégrer facilement avec des pipelines d’évaluation via `scikit-learn`

---

## 🛠️ Technologies utilisées

- Python 3.x
- TensorFlow / Keras
- NumPy
- scikit-learn

---
