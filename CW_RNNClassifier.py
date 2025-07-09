# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
# Suppress logging for SHAP
logging.getLogger("shap").setLevel(logging.ERROR)

import shap
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from classificationModels.ImplementationCW_RNN import CWRNNClassifier
from evaluationModels.evaluation_classification import ClassifierEvaluator

class Method_CW_RNN_Classifier:
    def __init__(self):
        self.best_cw_rnn = None
        self.explainer = None
        self.X_train_summary = None

    def preprocess_data(self, X):
        """
        Preprocess the data by converting it to a numeric format if necessary.

        Args:
            X: Input data, can be a DataFrame or numpy array.

        Returns:
            numpy array of processed data.
        """
        if isinstance(X, np.ndarray):
            return X  # Assuming it's already numeric

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)  # Convert to DataFrame if not already

        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'datetime64[ns]':
                # Convert datetime to seconds since epoch
                X_processed[col] = X_processed[col].astype('int64') / 10**9  
            elif X_processed[col].dtype == 'object':
                # Convert categorical data to numeric codes
                X_processed[col] = X_processed[col].astype('category').cat.codes  
            elif not np.issubdtype(X_processed[col].dtype, np.number):
                # Convert non-numeric data to float
                X_processed[col] = X_processed[col].astype(float)  

        return X_processed.values  # Return as numpy array

    def flatten_data(self, X):
        """
        Flatten the data to 2D array.

        Args:
            X: Input data.

        Returns:
            Flattened numpy array.
        """
        return X.reshape((X.shape[0], -1))

    def train_cw_rnn(self, X_train, y_train, input_shape, num_classes, epochs=10, n_iter=2, cv=5, random_state=42):
        """
        Train the CW-RNN model using RandomizedSearchCV.

        Args:
            X_train: Training data.
            y_train: Training labels.
            input_shape: Shape of the input data.
            num_classes: Number of classes for classification.
            epochs: Number of training epochs.
            n_iter: Number of iterations for RandomizedSearchCV.
            cv: Number of cross-validation folds.
            random_state: Random state for reproducibility.

        Returns:
            self
        """
        print("Veuillez patienter quelques instants...")

        # Preprocess the training data
        X_train = self.preprocess_data(X_train)

        # Ensure X_train is 3D (samples, timesteps, features)
        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, axis=1)
        
        # Initialize CW-RNN classifier
        cw_rnn = CWRNNClassifier(
            units=50,
            clock_periods=[10, 20, 30],
            input_shape=input_shape,
            num_classes=num_classes
        )

        # Define parameter distribution for RandomizedSearchCV
        param_dist = {
            'units': sp_randint(10, 100),
            'clock_periods': [[10, 20, 30], [5, 10, 15]],
        }

        # Perform randomized search for hyperparameter tuning
        random_search = RandomizedSearchCV(estimator=cw_rnn, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)

        self.best_cw_rnn = random_search.best_estimator_
        print(f"Le modèle CW-RNN a été entraîné avec les meilleurs hyperparamètres: {random_search.best_params_}.")

        # Flatten data for SHAP kmeans
        X_train_flattened = self.flatten_data(X_train)
        self.X_train_summary = shap.kmeans(X_train_flattened, 30).data

        # Initialize SHAP KernelExplainer
        self.explainer = shap.KernelExplainer(lambda x: self.best_cw_rnn.predict(x.reshape((-1, *input_shape))), self.X_train_summary)

        return self

    def predict(self, X_test):
        """
        Make predictions using the trained CW-RNN model.

        Args:
            X_test: Test data.

        Returns:
            Predicted labels.
        """
        if self.best_cw_rnn is None:
            raise ValueError("Le modèle n'a pas été entraîné. Veuillez appeler la méthode 'train_cw_rnn' d'abord.")
        else:
            print("La prédiction avec les données de test...")

        # Preprocess the test data
        X_test = self.preprocess_data(X_test)

        # Ensure X_test is 3D (samples, timesteps, features)
        if X_test.ndim == 2:
            X_test = np.expand_dims(X_test, axis=1)

        return self.best_cw_rnn.predict(X_test)

    def explain(self, X_instance):
        """
        Explain the predictions for a single instance using SHAP.

        Args:
            X_instance: Data instance to explain.

        Returns:
            SHAP values for the instance.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call 'train_cw_rnn' with training data first.")
        
        # Preprocess the instance data
        X_instance = self.preprocess_data(X_instance)

        # Ensure X_instance is 3D (samples, timesteps, features)
        if X_instance.ndim == 2:
            X_instance = np.expand_dims(X_instance, axis=1)

        # Flatten the instance data
        X_instance_flattened = self.flatten_data(X_instance)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_instance_flattened)
        return shap_values

    def summary_plot(self):
        """
        Generate a summary plot of SHAP values.

        Raises:
            ValueError: If the explainer is not fitted.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call 'train_cw_rnn' with training data first.")
        
        # Compute SHAP values for the training summary data
        shap_values = self.explainer.shap_values(self.X_train_summary)

        # Generate SHAP summary plot
        shap.summary_plot(shap_values, self.X_train_summary)

    def run_cw_rnn_classifier(self, X_train, y_train, X_test, y_test):
        """
        Run the full CW-RNN classification process: training, prediction, evaluation, and explanation.

        Args:
            X_train: Training data.
            y_train: Training labels.
            X_test: Test data.
            y_test: Test labels.
        """
        print("______________Entraînement du modèle CW-RNN______________")

        # Preprocess the training data
        X_train = self.preprocess_data(X_train)
        
        # Ensure X_train is 3D (samples, timesteps, features)
        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, axis=1)
            
        # Define the input shape and number of classes
        input_shape = (X_train.shape[1], X_train.shape[2])  # Assumes X_train is a 3D array
        num_classes = len(np.unique(y_train))  # Assumes y_train contains all class labels
        
        # Train the CW-RNN model
        self.train_cw_rnn(X_train, y_train, input_shape, num_classes)

        # Predict using the trained model
        y_pred = self.predict(X_test)

        # Print evaluation metrics
        print('_________________Evaluation Metrics_________________')
        evaluator = ClassifierEvaluator(y_test, y_pred)  # Ensure this class is defined elsewhere.
        evaluator.evaluation_metrics()
        
        # Explain the model predictions using SHAP
        print('_________________Explicabilité du Modèle CW-RNN avec SHAP_________________')
        print('Découvrez comment les différentes caractéristiques influencent les prédictions...')
        self.summary_plot()
