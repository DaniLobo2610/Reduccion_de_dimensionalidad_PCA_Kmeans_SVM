import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    silhouette_score
)

# ==========================
# Configuración
st.set_page_config(
    page_title="Clasificador MNIST",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Clasificador de Dígitos Manuscritos")

st.write(
    """
    Aplicación desarrollada utilizando:

    • PCA (Reducción de dimensionalidad)

    • K-Means (Clustering)

    • Support Vector Machine (Clasificación)

    REALIZADA POR ALBERTO DANIEL LOBO - 20211900125 - CLASE DE IA

    NOTA: Estimado usuario si quiere vivir toda la experiencia con este modelo le recomiendo tomar en cuenta lo siguiente: 

    • Despues de escoger los componentes a utilizar y entrenar el modelo; Si desea cambiar de componentes lo recomendable es volver a entrenar. 

    • Todo está explicado para mantener el orden y la compresión suya y mia

    • En la ultima parte donde se genera la imagen seleccionada por el modelo, en la parte de abajo hay un boton para mostrar resultados
    """
)

df = pd.read_csv("train.csv")

@st.cache_resource
def entrenar_modelo(n_componentes):

    # ==========================
    # Separar variables
    # ==========================

    X = df.drop("label", axis=1)
    y = df["label"]

    # ==========================
    # Escalamiento
    # ==========================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ==========================
    # PCA
    # ==========================

    pca = PCA(
        n_components=n_componentes,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)

    varianza = pca.explained_variance_ratio_.sum()

    # ==========================
    # K-Means
    # ==========================

    kmeans = KMeans(
        n_clusters=10,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_pca)

    silhouette = silhouette_score(
    X_pca,
    clusters
    )

    # ==========================
    # Train/Test
    # ==========================

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==========================
    # SVM
    # ==========================

    svm = SVC(kernel="rbf")

    svm.fit(X_train, y_train)

    predicciones = svm.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    reporte = classification_report(
        y_test,
        predicciones,
        output_dict=True
    )

    return (
        scaler,
        pca,
        kmeans,
        svm,
        X,
        y,
        X_pca,
        clusters,
        accuracy,
        reporte,
        silhouette,
        varianza
    )

st.header("Configuración del modelo")

n_componentes = st.slider(
    "Número de componentes principales (PCA)",
    min_value=30,
    max_value=60,
    value=30
)

if st.button("Entrenar Modelo"):

    with st.spinner("Entrenando modelo..."):

        (
            scaler,
            pca,
            kmeans,
            svm,
            X,
            y,
            X_pca,
            clusters,
            accuracy,
            reporte,
            silhouette,
            varianza
        ) = entrenar_modelo(n_componentes)

    st.success("Modelo entrenado correctamente.")

    
    st.session_state["modelo"] = {

        "scaler": scaler,

        "pca": pca,

        "kmeans": kmeans,

        "svm": svm,

        "X": X,

        "y": y,

        "X_pca": X_pca,

        "clusters": clusters,

        "accuracy": accuracy,

        "reporte": reporte,

        "silhouette": silhouette,

        "varianza": varianza

    }

if "modelo" in st.session_state:

    modelo = st.session_state["modelo"]

    st.header("Resultados del modelo")

    st.write("""
        Las siguientes métricas permiten evaluar el desempeño del clasificador SVM
        utilizando los datos transformados mediante PCA.
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{modelo['accuracy']*100:.2f}%"
        )

    with col2:
        st.metric(
            "Silhouette",
            f"{modelo['silhouette']:.3f}"
        )

    with col3:
        st.metric(
            "Varianza explicada",
            f"{modelo['varianza']*100:.2f}%"
        )
    
    st.divider()

    st.subheader("Reporte de Clasificación")

    reporte_df = pd.DataFrame(
        modelo["reporte"]
    ).transpose()

    st.dataframe(
    reporte_df.round(3),
    use_container_width=True,
    height=420
    )

    st.header("Visualización PCA y Clusters")

    df_pca = pd.DataFrame(
        modelo["X_pca"][:, :2],
        columns=["PC1", "PC2"]
    )

    df_pca["cluster"] = modelo["clusters"]

    st.info(
        f"""
        Se utilizarán **{n_componentes} componentes principales** para reducir la dimensionalidad
        de las imágenes antes de aplicar K-Means y SVM.
        """
    )


    st.write("""
    Cada punto representa una imagen del conjunto de datos MNIST.

    Los colores indican el **cluster** al que fue asignada cada imagen por el algoritmo K-Means.

    La posición de los puntos corresponde a la proyección de las imágenes utilizando las dos primeras componentes principales (PCA).
    """)

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

    st.success("""
        Conclusión:

        La proyección en dos dimensiones permite observar cómo K-Means agrupó
        las imágenes de acuerdo con sus características después de aplicar PCA.
        """)

    ax.set_title("Clusters obtenidos mediante PCA + K-Means")

    st.pyplot(fig)

    st.info("""
    Interpretación:

    • Cada punto corresponde a un dígito manuscrito.

    • Los colores representan los grupos encontrados automáticamente por K-Means.

    • Puntos cercanos poseen características similares.

    • Puntos alejados representan imágenes con patrones diferentes.
    """)

    st.divider()

    st.header("Clasificación de un Dígito")

    indice = st.number_input(
        "Índice de la imagen",
        min_value=0,
        max_value=len(modelo["X"]) - 1,
        value=0,
        step=1
    )

    imagen = modelo["X"].iloc[indice]
    etiqueta_real = modelo["y"].iloc[indice]


    st.subheader("Imagen seleccionada")

    col1, col2 = st.columns([1,3])

    with col1:

        fig, ax = plt.subplots(figsize=(2,2))

        ax.imshow(
            imagen.values.reshape(28,28),
            cmap="gray",
            interpolation="nearest"
        )

        ax.axis("off")

        st.pyplot(fig)

    with col2:

        st.write("""
        Esta es la imagen seleccionada del conjunto de datos MNIST
        que será clasificada por el modelo SVM.
        """)


    if st.button("Clasificar Imagen"):
        imagen_df = pd.DataFrame([imagen])

        imagen_escalada = modelo["scaler"].transform(imagen_df)

        imagen_pca = modelo["pca"].transform(imagen_escalada)


        prediccion = modelo["svm"].predict(imagen_pca)[0]


        st.subheader("Resultado de la Clasificación")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Etiqueta Real",
                int(etiqueta_real)
            )

        with col2:
            st.metric(
                "Predicción SVM",
                int(prediccion)
            )


        if prediccion == etiqueta_real:

            st.success("""
            ✅ El modelo clasificó correctamente el dígito.
            """)

        else:

            st.error("""
            ❌ El modelo no clasificó correctamente el dígito.
            """)


        st.info("""
        La imagen seleccionada fue transformada mediante PCA utilizando el número de componentes elegido y posteriormente clasificada por el modelo Support Vector Machine (SVM).
        """)


