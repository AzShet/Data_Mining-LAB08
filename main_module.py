# -*- coding: utf-8 -*-
"""main_module.py

Módulo principal para el Laboratorio 8: Clasificación con Árbol de Decisión y Bosques Aleatorios.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo

# Para ignorar advertencias
warnings.filterwarnings('ignore')

TARGET = 'Class'
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5

def preprocess_data(df: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series): # type: ignore
    """
    Realiza preprocesamiento completo de datos incluyendo:
    - Imputación de valores faltantes
    - Tratamiento de outliers
    - Estandarización de características numéricas
    - Balanceo de clases con SMOTE

    Args:
        df: DataFrame con los datos originales
        target: Nombre de la columna objetivo

    Returns:
        X: DataFrame con features procesadas
        y: Serie con la variable objetivo
    """
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputación de valores faltantes
    if X.isnull().sum().sum() > 0:
        if num_cols:
            imputer_num = SimpleImputer(strategy='mean')
            X[num_cols] = imputer_num.fit_transform(X[num_cols])
        if cat_cols:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

    # Tratamiento de outliers
    if num_cols:
        for col in num_cols:
            Q1, Q3 = X[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower_bound, upper_bound)

    # Estandarización de características numéricas
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Balanceo de clases con SMOTE
    class_counts = y.value_counts()
    min_prop = class_counts.min() / len(y)

    if min_prop < 0.5:
        smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X = pd.DataFrame(X_resampled, columns=X.columns)
        y = pd.Series(y_resampled, name=target)

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Divide los datos en train/test (75/25) y aplica stratify.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train, cv: int = CV_FOLDS) -> DecisionTreeClassifier:
    param_grid = {'max_depth': list(range(1, 11))}
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_random_forest(X_train, y_train, cv: int = CV_FOLDS) -> RandomForestClassifier:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return metrics
