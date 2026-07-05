import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import json

from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE LA APLICACIÓN

st.set_page_config(
    page_title="Clasificador MNIST",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Clasificador de Dígitos Manuscritos")

st.write("""
Aplicación desarrollada utilizando técnicas de Inteligencia Artificial.
         
## Realizada por: ALBERTO DANIEL LOBO - 20211900125

### Algoritmos implementados

• PCA (Principal Component Analysis)

• K-Means

• Support Vector Machine (SVM)

La aplicación permite comparar distintos modelos entrenados utilizando
10, 20, 30, 40 y 50 componentes principales.
""")


# ==========================================
# CARGAR DATASET

df = pd.read_csv("train.csv")

X = df.drop("label", axis=1)

y = df["label"]



# ==========================================
# CARPETA DE MODELOS

BASE_MODELOS = Path("modelos")


@st.cache_resource
def cargar_modelo(componentes):

    carpeta = BASE_MODELOS / f"modelo_{componentes}"

    modelo = {

        "scaler": joblib.load(carpeta / "scaler.pkl"),

        "pca": joblib.load(carpeta / "pca.pkl"),

        "kmeans": joblib.load(carpeta / "kmeans.pkl"),

        "svm": joblib.load(carpeta / "svm.pkl")

    }

    with open(carpeta / "metadata.json", encoding="utf-8") as f:

        modelo["metadata"] = json.load(f)

    modelo["clusters"] = pd.read_csv(
        carpeta / "mnist_clusters.csv"
    )

    modelo["metricas"] = pd.read_csv(
        carpeta / "svm_metricas.csv"
    )

    return modelo


# ==========================================
# CONFIGURACIÓN DEL MODELO

st.header("Configuración del Modelo")

st.write("""
Seleccione el número de componentes principales (PCA) con el que desea
trabajar. Cada opción corresponde a un modelo previamente entrenado.
""")


n_componentes = st.select_slider(
    "Número de componentes principales",
    options=[10, 20, 30, 40, 50],
    value=30
)


with st.spinner("Cargando modelo..."):

    modelo = cargar_modelo(n_componentes)

st.success(
    f"Se cargó correctamente el modelo entrenado con {n_componentes} componentes principales."
)


# ==========================================
# RESULTADOS DEL MODELO
st.header("Resultados del Modelo")

st.write("""
Las siguientes métricas corresponden al modelo previamente entrenado
utilizando la cantidad de componentes principales seleccionada.
""")

metadata = modelo["metadata"]

metricas = modelo["metricas"]




col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Accuracy",
        f"{metricas['Accuracy'][0]*100:.2f}%"
    )

with col2:

    st.metric(
        "Silhouette",
        f"{metricas['Silhouette'][0]:.3f}"
    )

with col3:

    st.metric(
        "Varianza explicada",
        f"{metadata['varianza_explicada']*100:.2f}%"
    )

st.divider()

st.subheader("Información del Modelo")

info = pd.DataFrame({

    "Propiedad":[

        "Nombre del modelo",

        "Dataset",

        "Componentes PCA",

        "Número de Clusters",

        "Kernel_SVM"

    ],

    "Valor":[

        "MNIST PCA + K-Means + SVM",

        "MNIST",

        metadata["componentes_pca"],

        metadata["n_clusters"],

        metadata["kernel_svm"]

    ]

})

st.dataframe(
    info,
    width="stretch"
)


st.divider()

st.header("Visualización PCA + K-Means")


st.write("""
Cada punto representa una imagen del conjunto de datos MNIST.

Las imágenes fueron proyectadas mediante PCA a dos dimensiones.

Posteriormente K-Means agrupó automáticamente las imágenes en diez grupos.
""")


df_pca = modelo["clusters"]

fig, ax = plt.subplots(figsize=(10,7))

sns.scatterplot(

    data=df_pca,

    x="PC1",

    y="PC2",

    hue="cluster",

    palette="tab10",

    s=20,

    alpha=0.7,

    ax=ax

)

ax.set_title("Clusters generados mediante PCA + K-Means")

st.pyplot(fig)
plt.close(fig)

st.info("""
Interpretación

• Cada punto representa una imagen.

• Los colores representan el grupo encontrado por K-Means.

• Imágenes cercanas poseen características similares.

• El gráfico utiliza únicamente las dos primeras componentes principales para facilitar la visualización.
""")

# ==========================================
# CLASIFICACIÓN DE UN DÍGITO

st.divider()

st.header("Clasificación de un Dígito")

st.write("""
Seleccione una imagen del conjunto de datos MNIST.

El modelo transformará la imagen mediante PCA y posteriormente
realizará la clasificación utilizando Support Vector Machine (SVM).
""")


indice = st.slider(
    "Seleccione una imagen",
    min_value=0,
    max_value=len(df)-1,
    value=0
)


imagen = X.iloc[indice]

etiqueta_real = y.iloc[indice]


col1, col2 = st.columns([1,2])

with col1:

    st.subheader("Imagen")

    fig, ax = plt.subplots(figsize=(2.5,2.5))

    ax.imshow(
        imagen.values.reshape(28,28),
        cmap="gray"
    )

    ax.axis("off")

    st.pyplot(fig)


with col2:

    st.subheader("Información")

    st.write(f"Índice seleccionado: **{indice}**")

    st.write(f"Etiqueta real: **{etiqueta_real}**")


if st.button("Clasificar Imagen"):

    imagen_df = pd.DataFrame([imagen])

    imagen_escalada = modelo["scaler"].transform(
        imagen_df
    )

    imagen_pca = modelo["pca"].transform(
        imagen_escalada
    )

    prediccion = modelo["svm"].predict(
        imagen_pca
    )[0]

    st.subheader("Resultado")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Etiqueta Real",
            int(etiqueta_real)
        )

    with c2:

        st.metric(
            "Predicción SVM",
            int(prediccion)
        )

    if prediccion == etiqueta_real:

        st.success(
            "✅ El modelo clasificó correctamente la imagen."
        )

    else:

        st.error(
            "❌ El modelo no clasificó correctamente la imagen."
        )

    st.divider()

    st.caption("""
    Proyecto desarrollado para la asignatura de Inteligencia Artificial.

    Modelo basado en PCA + K-Means + Support Vector Machine utilizando el conjunto de datos MNIST.
    """)
