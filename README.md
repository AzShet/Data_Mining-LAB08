# Clasificación con Árbol de Decisión y Bosques Aleatorios

Este repositorio contiene el código y los tests, donde se implementa un modelo de clasificación utilizando Árboles de Decisión y Bosques Aleatorios. El proyecto se basa en el conjunto de datos de cáncer de mama de Wisconsin. Se aplican técnicas de clasificación utilizando modelos de Árbol de Decisión y Bosques Aleatorios, incluyendo un pipeline completo desde la limpieza de datos hasta la evaluación comparativa de modelos, manteniendo buenas prácticas de programación y pruebas automatizadas con `pytest`.

## Contenido del Repositorio

- **LAB08-RUELAS.ipynb**: El archivo original en formato Jupyter Notebook.
- **main_module.py**: Archivo que contiene las funciones y la lógica principal del laboratorio.
- **test_lab08_ruelas.py**: Archivo que contiene los tests para verificar la funcionalidad del código.

## 🧩 Funcionalidades

- Limpieza e imputación de datos faltantes.
- Detección y tratamiento de outliers (IQR Capping).
- Estandarización de variables numéricas.
- Balanceo de clases con SMOTE.
- Entrenamiento con Árbol de Decisión y Bosque Aleatorio (GridSearchCV).
- Evaluación con métricas: accuracy, precision, recall, f1-score.
- Pruebas automatizadas con `pytest`.

## Proceso de Conversión

1. **Creación de `main_module.py`**:
   - Se creó un nuevo archivo llamado `main_module.py`, que contiene todas las funciones necesarias para el laboratorio. Este archivo incluye:
     - **Funciones de Preprocesamiento**: `preprocess_data` para manejar valores faltantes, outliers, estandarización y balanceo de clases.
     - **Funciones de División de Datos**: `split_data` para dividir el conjunto de datos en entrenamiento y prueba.
     - **Funciones de Entrenamiento**: `train_decision_tree` y `train_random_forest` para entrenar los modelos.
     - **Función de Evaluación**: `evaluate_model` para calcular métricas de rendimiento.

2. **Creación de `test_lab08_ruelas.py`**:
   - Se creó un archivo de pruebas llamado `test_lab08_ruelas.py`, que utiliza `pytest` para verificar la funcionalidad del código. Este archivo incluye:
     - **Fixtures**: Para crear un DataFrame de muestra con un desbalance de clases.
     - **Tests**: Para verificar que las funciones de preprocesamiento, división de datos, entrenamiento de modelos y evaluación funcionen correctamente.

## 📁 Estructura del Proyecto

```
├── main_module.py        # Contiene funciones principales de preprocesamiento y modelado
├── test_lab08_ruelas.py  # Archivo de tests con Pytest
├── README.md             # Documentación del repositorio
└── requirements.txt      # (opcional) Librerías necesarias
```

## Uso del Laboratorio

### Requisitos

* Python 3.8 o superior

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy scikit-learn imbalanced-learn ucimlrepo pytest
```

También puedes crear un archivo `requirements.txt` con:

```txt
pandas
numpy
scikit-learn
imbalanced-learn
ucimlrepo
pytest
```

## 📊 Resultados Esperados

* Se muestran métricas detalladas por clase (precision, recall, f1).
* Comparación clara entre Árbol y Random Forest.
* Todos los tests de `pytest` deben pasar exitosamente.

## 👨‍💻 Autor

* **César Diego Ruelas Flores**  
  Estudiante de Big Data y Ciencia de Datos - TECSUP  
  [cesar.ruelas@tecsup.edu.pe](mailto:cesar.ruelas@tecsup.edu.pe)

---

> “La calidad del código no está en lo que hace, sino en cómo lo hace.”
