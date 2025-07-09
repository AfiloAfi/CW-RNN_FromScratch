"""
Ce code implémente une version spécialisée d'un Réseau Récurrent à Horloge (CW-RNN) en utilisant TensorFlow et Keras. 
Le CW-RNN est une variante des réseaux récurrents standards, où certaines unités sont mises à jour à des intervalles 
de temps réguliers (périodes d'horloge), tandis que les autres unités restent fixes.

Principes suivis pour implémenter la couche CW-RNN :

1. **Cellule CW-RNN** :
   - La classe `CWRNNCell` hérite de `SimpleRNNCell` de Keras et ajoute la fonctionnalité de mise à jour périodique.
   - Lors de chaque appel, la cellule vérifie si le nombre actuel de pas de temps (`self.step`) est un multiple de 
     la période de l'horloge (`self.clock_period`). Si oui, elle met à jour son état ; sinon, elle conserve son état 
     précédent.
   - Cette mise à jour périodique permet de réduire les calculs et peut aider à capturer des dépendances à plus long terme.

2. **Modèle CW-RNN** :
   - La classe `CWRNN` est construite en utilisant une liste de cellules CW-RNN (`CWRNNCell`).
   - Elle initialise les états des cellules et passe les entrées à travers chaque cellule séquentiellement, 
     tout en mettant à jour les états selon la logique définie dans `CWRNNCell`.

3. **Classificateur CW-RNN** :
   - La classe `CWRNNClassifier` encapsule le modèle CW-RNN dans une structure de classificateur compatible avec 
     `scikit-learn`, facilitant ainsi son utilisation avec des outils d'évaluation de modèles comme `RandomizedSearchCV`.
   - Le modèle est compilé avec un optimiseur, une fonction de perte et des métriques.
   - Les méthodes `fit`, `predict`, `score`, `save_model`, et `load_model` sont définies pour permettre l'entraînement, 
     la prédiction, l'évaluation et la gestion du modèle.

Ce code offre une solution efficace pour la modélisation des séquences temporelles en utilisant le concept de mise à jour 
périodique des unités dans un réseau récurrent, améliorant ainsi l'efficacité computationnelle et la capture des dépendances 
à long terme.
"""

import tensorflow as tf
from tensorflow.keras.layers import SimpleRNNCell
from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np

# Définition de la cellule CW-RNN, qui est une variante de SimpleRNNCell
class CWRNNCell(SimpleRNNCell):
    def __init__(self, units, clock_period, **kwargs):
        """
        Initialisation de la cellule CW-RNN.
        
        Args:
            units (int): Dimensionalité de l'espace de sortie.
            clock_period (int): Nombre de pas de temps avant la mise à jour de l'état de la cellule.
        """
        super(CWRNNCell, self).__init__(units, **kwargs)
        self.clock_period = clock_period  # Période de l'horloge pour la cellule
        self.step = 0  # Initialisation du compteur de pas de temps

    def call(self, inputs, states, training=None):
        """
        Méthode d'appel de la cellule CW-RNN.
        
        Args:
            inputs (tensor): Tensor d'entrée.
            states (list): Liste des tenseurs d'état.
            training (bool): Indique si la couche doit fonctionner en mode entraînement ou inférence.
        
        Returns:
            outputs (tensor): Tenseur de sortie.
            new_states (list): Liste des nouveaux tenseurs d'état.
        """
        self.step += 1  # Incrémentation du compteur de pas de temps
        if self.step % self.clock_period == 0:
            # Mise à jour de l'état de la cellule à chaque période de l'horloge
            return super(CWRNNCell, self).call(inputs, states, training)
        else:
            # Sinon, maintenir l'état précédent
            return states[0], states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Méthode pour obtenir l'état initial de la cellule.
        
        Args:
            inputs: Ignoré dans cette implémentation.
            batch_size: Taille du lot.
            dtype: Type de données de l'état initial.
        
        Returns:
            Liste des tenseurs d'état initiaux.
        """
        return [tf.zeros((batch_size, self.units), dtype=dtype)]

    def get_config(self):
        """
        Renvoie la configuration de la cellule. Utilisé pour la sérialisation.
        """
        config = super(CWRNNCell, self).get_config()
        config.update({'clock_period': self.clock_period})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Crée une cellule à partir de sa configuration. Utilisé pour la désérialisation.
        """
        return cls(**config)

# Définition du modèle CW-RNN
class CWRNN(Model):
    def __init__(self, cells, **kwargs):
        """
        Initialisation du modèle CW-RNN.
        
        Args:
            cells (list): Liste des cellules CW-RNN.
        """
        super(CWRNN, self).__init__(**kwargs)
        self.cells = cells  # Liste des cellules CW-RNN

    def call(self, inputs, training=None):
        """
        Méthode d'appel du modèle CW-RNN.
        
        Args:
            inputs (tensor): Tenseur d'entrée.
            training (bool): Indique si la couche doit fonctionner en mode entraînement ou inférence.
        
        Returns:
            outputs (tensor): Tenseur de sortie.
        """
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
        # Initialisation des états des cellules
        states = [cell.get_initial_state(batch_size=batch_size, dtype=dtype) for cell in self.cells]
        outputs = inputs
        for i, cell in enumerate(self.cells):
            outputs, states[i] = cell(outputs, states=states[i], training=training)
        return outputs

    def get_config(self):
        """
        Renvoie la configuration du modèle. Utilisé pour la sérialisation.
        """
        config = super(CWRNN, self).get_config()
        config.update({
            'cells': [tf.keras.layers.serialize(cell) for cell in self.cells]
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Crée un modèle à partir de sa configuration. Utilisé pour la désérialisation.
        """
        cells = [tf.keras.layers.deserialize(cell_config) for cell_config in config.pop('cells')]
        return cls(cells, **config)

# Définition du classificateur CW-RNN
class CWRNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, units, clock_periods, input_shape, num_classes, optimizer='adam', loss_fn='sparse_categorical_crossentropy', epochs=10, batch_size=32, validation_data=None):
        """
        Initialisation du classificateur CW-RNN.
        
        Args:
            units (int): Dimensionalité de l'espace de sortie.
            clock_periods (list): Liste des périodes de l'horloge pour chaque cellule CW-RNN.
            input_shape (tuple): Forme des données d'entrée.
            num_classes (int): Nombre de classes pour la couche de sortie.
            optimizer: Optimiseur à utiliser pendant l'entraînement.
            loss_fn: Fonction de perte à utiliser pendant l'entraînement.
            epochs (int): Nombre d'époques pour entraîner le modèle.
            batch_size (int): Taille du lot pour l'entraînement.
            validation_data (tuple): Données de validation sous la forme (X_val, y_val).
        """
        self.units = units
        self.clock_periods = clock_periods
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.model = self._build_model()  # Construction du modèle CW-RNN

    def _build_model(self):
        """
        Méthode pour construire le modèle CW-RNN.
        """
        # Création des cellules CW-RNN
        cells = [CWRNNCell(self.units, period) for period in self.clock_periods]
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = CWRNN(cells)(inputs)
        x = tf.keras.layers.Flatten()(x)  # Aplatir la sortie pour la couche dense
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train):
        """
        Méthode pour entraîner le classificateur CW-RNN.
        
        Args:
            X_train (array-like): Données d'entraînement.
            y_train (array-like): Étiquettes d'entraînement.
        """
        # Reshape les données d'entrée si nécessaire
        if X_train.ndim == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=self.validation_data)

    def predict(self, X_test):
        """
        Méthode pour effectuer des prédictions à l'aide du classificateur CW-RNN.
        
        Args:
            X_test (array-like): Données de test.
        
        Returns:
            y_pred (array-like): Étiquettes prédites.
        """
        # Reshape les données de test si nécessaire
        if X_test.ndim == 2:
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        return tf.argmax(self.model.predict(X_test), axis=1).numpy()

    def score(self, X, y):
        """
        Méthode pour calculer le score (par exemple, l'exactitude) du modèle sur les données de test.
        
        Args:
            X (array-like): Données d'entrée.
            y (array-like): Étiquettes de sortie.
        
        Returns:
            score (float): Score du modèle.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def save_model(self, filepath):
        """
        Enregistrer le modèle dans un fichier.
        
        Args:
            filepath (str): Chemin du fichier.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Charger le modèle depuis un fichier.
        
        Args:
            filepath (str): Chemin du fichier.
        """
        self.model = tf.keras.models.load_model(filepath, custom_objects={'CWRNNCell': CWRNNCell, 'CWRNN': CWRNN})
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])
