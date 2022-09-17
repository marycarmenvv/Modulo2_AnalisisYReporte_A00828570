# Modulo2_AnalisisYReporte_A00828570

# Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)
# Nombre de los proyectos a tomar en cuenta del repositorio:
* Final_Modulo2_AnalisisYReporte_A00828570.ipynb
* Final_Modulo2_AnalisisYReporte_A00828570.pdf
* final_Modulo2_AnalisisYReporte_A00828570.py

Concentración: Inteligencia Artificial Avanzada para la ciencia de datos

Descripción de la entrega: Entregable: Análisis y Reporte sobre el desempeño del modelo.

Escoge una de las 2 implementaciones que tengas y genera un análisis sobre su desempeño en un set de datos. Este análisis lo deberás documentar en un reporte con indicadores claros y gráficas comparativas que respalden tu análisis.
El análisis debe de contener los siguientes elementos:
Separación y evaluación del modelo con un conjunto de prueba y un conjunto de validación (Train/Test/Validation).
- Diagnóstico y explicación el grado de bias o sesgo: bajo medio alto
- Diagnóstico y explicación el grado de varianza: bajo medio alto
- Diagnóstico y explicación el nivel de ajuste del modelo: underfitt fitt overfitt
- Basándote en lo encontrado en tu análisis utiliza técnicas de regularización o ajuste de parámetros para mejorar el desempeño de tu modelo y documenta en tu reporte cómo mejoró este.

# Modelo de aprendizaje supervisado implementado: Decision Trees

Data set utilizado:
*  wine.data (Obteniéndose un Accuracy: 0.97)

# Librerías utilizadas:
- import numpy as np
- import pandas as pd
- import seaborn as sns

- from sklearn.model_selection import train_test_split
- from sklearn import metrics
- from sklearn.metrics import confusion_matrix
- from sklearn.metrics import accuracy_score

- from sklearn.tree import DecisionTreeClassifier
- from sklearn import tree
- from sklearn import preprocessing
- from IPython.display import Image
- import pydotplus
- import matplotlib.pyplot as plt
- from matplotlib import pyplot
