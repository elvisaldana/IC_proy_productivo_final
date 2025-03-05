import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from supabase_api import fetch_data_from_supabase
import numpy as np

def clustering_proveedores():
    st.header("🔍 Segmentación de Proveedores - Análisis Estratégico")

    # **Instrucciones para el Analista de Compras**
    st.markdown("""
    ### 📌 **Guía para el Analista de Compras**
    
    Esta herramienta permite **segmentar a los proveedores** según su **frecuencia de compra**, **monto total comprado** y **tiempo promedio de entrega**. 
    
    #### 🔹 **¿Cómo utilizar este módulo?**
    1. **Seleccione el método de clustering:** Puede elegir entre **K-Means** o **DBSCAN**.
    2. **Ajuste los parámetros:** 
        - Para **K-Means**, elija el número de clusters óptimo usando el "Método del Codo".
        - Para **DBSCAN**, ajuste los valores de `eps` y `min_samples` para mejorar la segmentación.
    3. **Analice los resultados:**
        - Revise el **Silhouette Score** para evaluar la calidad de los clusters.
        - Utilice el **gráfico de dispersión PCA** para visualizar los proveedores en dos dimensiones.
        - Filtre proveedores por cluster para un análisis más detallado.
    4. **Descargue los resultados en `.csv`** para compartir con su equipo.
    
    #### 📊 **¿Cómo interpretar los Clusters?**
    - **Cluster 0:** Proveedores con alta frecuencia de compra y montos elevados. **Aliados estratégicos**.
    - **Cluster 1:** Proveedores con baja frecuencia de compra y montos bajos. **Posibles optimizaciones**.
    - **Cluster 2:** Proveedores con tiempos de entrega inconsistentes. **Potenciales riesgos logísticos**.
    
    📢 **Sugerencia:** Utilice estos insights para negociar mejores términos con los proveedores clave o buscar alternativas para los que presentan problemas de entrega.
    """)

    # Cargar datos desde Supabase
    table_name = "vista_segmentacion_proveedores"
    try:
        data = fetch_data_from_supabase(table_name)
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Error al cargar datos desde Supabase: {e}")
        return

    if df.empty:
        st.warning("⚠️ No hay datos disponibles para segmentación.")
        return

    # Seleccionar características para clustering
    features = ["frecuencia_compra", "monto_total_compras", "tiempo_entrega_promedio"]
    df_cluster = df[features]

    # Calcular desviación estándar del tiempo de entrega
    df["desviacion_tiempo_entrega"] = df["tiempo_entrega_promedio"].apply(lambda x: np.std(x) if pd.notnull(x) else 0)

    # Escalar datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)

    # Método del Codo para seleccionar el número óptimo de clusters
    st.subheader("📊 Método del Codo para Seleccionar el Número de Clusters")
    
    inertia = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    fig_elbow = px.line(x=list(K_range), y=inertia, markers=True, title="Método del Codo - Selección de K")
    fig_elbow.update_layout(xaxis_title="Número de Clusters (K)", yaxis_title="Inercia")
    st.plotly_chart(fig_elbow)

    # Seleccionar algoritmo de clustering
    algoritmo = st.radio("🔢 Seleccione el algoritmo de clustering:", ["K-Means", "DBSCAN"])

    if algoritmo == "K-Means":
        # Seleccionar número de clusters
        num_clusters = st.slider("Seleccione el número de clusters:", min_value=2, max_value=6, value=3, step=1)

        # Aplicar K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(df_scaled)

        # Calcular Silhouette Score
        silhouette_avg = silhouette_score(df_scaled, df["cluster"])
        st.write(f"**📏 Silhouette Score:** {silhouette_avg:.3f} (Cercano a 1 significa clusters bien separados)")

    elif algoritmo == "DBSCAN":
        # Parámetros de DBSCAN
        eps_val = st.slider("Seleccione el valor de eps para DBSCAN:", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples_val = st.slider("Seleccione el valor de min_samples:", min_value=2, max_value=10, value=5, step=1)

        # Aplicar DBSCAN Clustering
        dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        df["cluster"] = dbscan.fit_predict(df_scaled)

        # Calcular Silhouette Score (solo si hay más de 1 cluster válido)
        if len(set(df["cluster"])) > 1:
            silhouette_avg = silhouette_score(df_scaled, df["cluster"])
            st.write(f"**📏 Silhouette Score:** {silhouette_avg:.3f} (Cercano a 1 significa clusters bien separados)")
        else:
            st.warning("⚠️ DBSCAN detectó un solo cluster. Ajuste los parámetros eps y min_samples.")

    # Visualización de Clusters con PCA
    st.subheader("📍 Visualización de Clusters con PCA")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)
    df["PCA_1"] = X_pca[:, 0]
    df["PCA_2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=df["PCA_1"], y=df["PCA_2"], hue=df["cluster"], palette='viridis', ax=ax)
    ax.set_title("Clusters en Espacio PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

    # Filtro interactivo para analizar proveedores por cluster
    st.subheader("🎯 Filtrar Proveedores por Cluster")
    cluster_selected = st.selectbox("Seleccione un cluster para analizar:", df["cluster"].unique())

    # Mostrar información de los proveedores en el cluster seleccionado
    st.subheader(f"📌 Proveedores en Cluster {cluster_selected}")
    st.dataframe(df[df["cluster"] == cluster_selected][["nombre_proveedor", "frecuencia_compra", "monto_total_compras", "tiempo_entrega_promedio", "desviacion_tiempo_entrega", "cluster"]])

    # Botón para descargar los resultados
    st.subheader("📥 Exportar Resultados")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV 📥",
        data=csv,
        file_name="segmentacion_proveedores.csv",
        mime="text/csv",
    )
