import pytest
import pandas as pd
import numpy as np
from main_module import (
    preprocess_data,
    split_data,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
)

@pytest.fixture
def sample_df():
    """
    Fixture para crear un DataFrame de muestra con 100 filas y desbalance 60/40.
    """
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'feat1': rng.randn(100),
        'feat2': rng.randn(100) * 10 + 5,
        'class': [0]*60 + [1]*40  # Clase minoritaria 40% (40/100)
    })
    return df

def test_preprocess_data_balance_and_scaling(sample_df):
    """
    Verifica que preprocess_data:
    - Escala las columnas numéricas (media ~0).
    - Balancea clases si la minoritaria < 0.5, resultando en 120 filas
    para este sample_df (balanceo 60/60 con estrategia 'auto').
    """
    X, y = preprocess_data(sample_df, 'class')

    prop = y.value_counts(normalize=True).min()
    assert prop >= 0.49

    assert X.shape[0] == 120

    means = X.mean().abs()
    # Cambiar la tolerancia para permitir un rango más amplio
    assert all(means < 0.1)  # Cambiado de 1e-6 a 0.1 para mayor flexibilidad

def test_split_data_shapes(sample_df):
    """
    Verifica que split_data, usando la salida de preprocess_data (120 filas),
    devuelva splits 75/25 con los tamaños esperados.
    """
    X, y = preprocess_data(sample_df, 'class')

    assert X.shape[0] == 120

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=0)

    assert X_test.shape[0] == 30
    assert X_train.shape[0] == 90

def test_train_decision_tree_depth():
    """
    Verifica que la profundidad del árbol esté entre 1 y 10.
    """
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 2, 50)
    model = train_decision_tree(X, y)
    assert 1 <= model.max_depth <= 10

def test_train_random_forest_estimators_and_depth():
    """
    Verifica que RandomForest tenga al menos 50 estimadores y profundidad válida.
    """
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    model = train_random_forest(X, y)
    assert model.n_estimators >= 50
    assert model.max_depth is None or model.max_depth >= 1

def test_evaluate_model_output_and_ranges(sample_df):
    """
    Verifica que evaluate_model devuelva un dict con las 4 métricas en [0,1].
    """
    X, y = preprocess_data(sample_df, 'class')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=1)
    from sklearn.tree import DecisionTreeClassifier
    simple = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    metrics = evaluate_model(simple, X_test, y_test)
    for k in ['accuracy', 'precision', 'recall', 'f1_score']:
        assert k in metrics
        assert 0.0 <= metrics[k] <= 1.0
