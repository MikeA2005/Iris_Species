# Integrantes
- Miguel Ayala
- Jose Romero
- Carlos Torres

# Proyecto Final - Clasificación de Especies de Iris

Este proyecto corresponde al curso de **Data Mining** en la Universidad de la Costa.  
El objetivo es entrenar un modelo de clasificación para predecir la especie de una flor Iris a partir de sus medidas.

## Funcionalidades
- Entrenamiento de un modelo de **Random Forest**.
- Visualización de métricas: Accuracy, Precision, Recall, F1.
- Panel interactivo para ingresar medidas y obtener la especie predicha.
- Gráfico 3D con la nueva muestra posicionada respecto al dataset.
- Histogramas de las características.

## Instalación
Clona el repositorio y asegúrate de tener Python 3.9+ instalado.

```bash
git clone https://github.com/tuusuario/iris-classification.git
cd iris-classification
pip install -r requirements.txt
streamlit run proyect.py