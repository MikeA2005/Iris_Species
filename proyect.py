import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

# --- Cargar dataset ---
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# --- Dividir datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entrenar modelo ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Métricas ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

# --- Interfaz Streamlit ---
st.title("Clasificación de Especies de Iris")
st.write("Proyecto final de Data Mining - Universidad de la Costa")

st.subheader("Integrantes")
st.write("Miguel Ayala")
st.write("Jose Romero")
st.write("Carlos Torres")

st.subheader("Métricas del modelo")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")

st.subheader("Predicción de nueva muestra")
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2)

new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(new_sample)[0]
pred_species = iris.target_names[prediction]

st.write(f"La especie predicha es: **{pred_species}**")

# --- Visualización 3D ---
st.subheader("Visualización 3D")
fig = px.scatter_3d(
    X, x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
    color=y.map(lambda i: iris.target_names[i]),
    title="Distribución de especies Iris"
)
fig.add_scatter3d(
    x=[sepal_length], y=[sepal_width], z=[petal_length],
    mode="markers", marker=dict(size=6, color="red"), name="Nueva muestra"
)
st.plotly_chart(fig)

# --- Visualizaciones adicionales ---
st.subheader("Histogramas de características")
st.bar_chart(X)