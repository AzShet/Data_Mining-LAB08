# Laboratorio 8: Clasificación con Árbol de Decisión y Bosques Aleatorios

Este repositorio contiene el código y los tests para el Laboratorio 8, donde se implementa un modelo de clasificación utilizando Árboles de Decisión y Bosques Aleatorios. El laboratorio se basa en el conjunto de datos de cáncer de mama de Wisconsin.

## Contenido del Repositorio

- **LAB08-RUELAS.ipynb**: El archivo original en formato Jupyter Notebook.
- **lab08_ruelas_SIN_CORREGIR.py**: Versión convertida del notebook a un archivo Python (.py) que contiene el código sin correcciones.
- **main_module.py**: Archivo que contiene las funciones y la lógica principal del laboratorio.
- **test_lab08_ruelas.py**: Archivo que contiene los tests para verificar la funcionalidad del código.

## Proceso de Conversión

1. **Conversión de Jupyter Notebook a Python**:
   - Se tomó el archivo `LAB08-RUELAS.ipynb` y se convirtió a un archivo Python llamado `lab08_ruelas_SIN_CORREGIR.py`. Este archivo contenía el código original sin correcciones.

2. **Corrección de Errores**:
   - Se identificaron y corrigieron errores en el archivo `lab08_ruelas_SIN_CORREGIR.py`, asegurando que el código funcionara correctamente y que los tests pudieran ejecutarse sin fallos.

3. **Creación de `main_module.py`**:
   - Se creó un nuevo archivo llamado `main_module.py`, que contiene todas las funciones necesarias para el laboratorio. Este archivo incluye:
     - **Funciones de Preprocesamiento**: `preprocess_data` para manejar valores faltantes, outliers, estandarización y balanceo de clases.
     - **Funciones de División de Datos**: `split_data` para dividir el conjunto de datos en entrenamiento y prueba.
     - **Funciones de Entrenamiento**: `train_decision_tree` y `train_random_forest` para entrenar los modelos.
     - **Función de Evaluación**: `evaluate_model` para calcular métricas de rendimiento.

4. **Creación de `test_lab08_ruelas.py`**:
   - Se creó un archivo de pruebas llamado `test_lab08_ruelas.py`, que utiliza `pytest` para verificar la funcionalidad del código. Este archivo incluye:
     - **Fixtures**: Para crear un DataFrame de muestra con un desbalance de clases.
     - **Tests**: Para verificar que las funciones de preprocesamiento, división de datos, entrenamiento de modelos y evaluación funcionen correctamente.

## Uso del Laboratorio

### Requisitos

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy scikit-learn imbalanced-learn ucimlrepo pytest
